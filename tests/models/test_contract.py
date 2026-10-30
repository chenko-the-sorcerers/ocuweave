import unittest
from dataclasses import fields
from ocuweave.models.ocuweave import ModelInputs
class ContractTests(unittest.TestCase):
    def test_supervision_is_not_a_model_input(self):
        names = {field.name for field in fields(ModelInputs)}
        self.assertNotIn("targets", names)
        self.assertNotIn("lesion_masks", names)
