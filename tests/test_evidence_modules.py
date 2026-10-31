import unittest
import torch
from ocuweave.models.ocuweave import OcuWeave, ModelInputs
from ocuweave.models.language.notes import tokenize_notes, NoteEncoder
from ocuweave.models.clinical.structured import ClinicalEncoder
from ocuweave.models.fusion.evidence import EvidenceFusion


class EvidenceModules(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_missing_modalities_match_image_only_and_ignore_payloads(self):
        model = OcuWeave(
            clinical_features=2, width=8, dim=16, queries=2, max_note_length=16
        ).eval()
        x = torch.rand(2, 3, 32, 32)
        absent = model(ModelInputs(x))
        torch.testing.assert_close(
            absent.predictions, model.vision(x), rtol=1e-5, atol=1e-6
        )
        contaminated = model(
            ModelInputs(
                x,
                note_ids=torch.full((2, 3), 999, dtype=torch.long),
                note_valid=torch.zeros(2, 3, dtype=torch.bool),
                clinical_values=torch.full((2, 2), float("nan")),
                clinical_observed=torch.zeros(2, 2, dtype=torch.bool),
            )
        )
        torch.testing.assert_close(absent.predictions, contaminated.predictions)
        self.assertFalse(contaminated.text_evidence["available"].any())
        self.assertTrue(
            torch.isfinite(contaminated.uncertainty["binary_entropy"]).all()
        )
        self.assertFalse(contaminated.uncertainty["calibrated"])

    def test_observed_evidence_has_gradients_and_no_target_inputs(self):
        model = OcuWeave(
            clinical_features=2, width=8, dim=16, queries=2, max_note_length=16
        )
        ids, valid = tokenize_notes(["shape one", "shape two"], 16)
        result = model(
            ModelInputs(
                torch.rand(2, 3, 32, 32),
                note_ids=ids,
                note_valid=valid,
                clinical_values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                clinical_observed=torch.ones(2, 2, dtype=torch.bool),
            )
        )
        result.predictions.mean().backward()
        self.assertGreater(float(model.language.embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.clinical.network[0].weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "Encode raw notes"):
            model(ModelInputs(torch.rand(1, 3, 32, 32), clinical_note="raw"))

    def test_training_only_normalization_and_unknown_value_invariance(self):
        encoder = ClinicalEncoder(2, 8)
        values = torch.tensor([[1.0, float("nan")], [3.0, 7.0]])
        observed = torch.isfinite(values)
        with self.assertRaisesRegex(ValueError, "training"):
            encoder.fit_statistics(values, observed, splits=["train", "validation"])
        encoder.fit_statistics(values, observed, splits=["train", "train"])
        torch.testing.assert_close(encoder.mean, torch.tensor([2.0, 7.0]))
        torch.testing.assert_close(encoder.scale, torch.tensor([1.0, 1.0]))
        altered = values.clone()
        altered[0, 1] = 1e30
        torch.testing.assert_close(
            encoder(values, observed)[0], encoder(altered, observed)[0]
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            encoder(values, torch.ones_like(observed))

    def test_token_padding_and_fusion_availability(self):
        ids, valid = tokenize_notes([None, " ", "é"], 4)
        self.assertEqual(ids[2, :2].tolist(), [196, 170])
        self.assertFalse(valid[:2].any())
        encoded, available = NoteEncoder(16, 4)(ids, valid)
        self.assertTrue(torch.isfinite(encoded).all())
        self.assertEqual(available.tolist(), [False, False, True])
        tokens = torch.full((2, 2, 16), float("nan"))
        mask = torch.zeros(2, 2, dtype=torch.bool)
        fused, _ = EvidenceFusion(16)(tokens, mask)
        self.assertEqual(float(fused.detach().abs().sum()), 0)
        mask[0, 0] = True
        with self.assertRaisesRegex(ValueError, "finite"):
            EvidenceFusion(16)(tokens, mask)


if __name__ == "__main__":
    unittest.main()
