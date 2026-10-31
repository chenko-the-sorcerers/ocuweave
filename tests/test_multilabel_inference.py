import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ocuweave.data.adapters.idrid import CLASSES
from ocuweave.models.multilabel import MultiLabelLQNet
from ocuweave.inference.multilabel import predict_lesions,save_lesion_prediction


class MultiLabelInferenceChecks(unittest.TestCase):
    def test_constant_logits_restore_all_channels_and_preserve_existing_outputs(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=MultiLabelLQNet()
            with torch.no_grad():
                for p in model.parameters():p.zero_()
                for head,bias in zip(model.heads,(-2.,0.,2.,4.)):head.bias.fill_(bias)
            checkpoint=root/'model.pt'
            torch.save({'model':model.state_dict(),'local_detail':True,'size':192,
                        'protocol':{'protocol':'idrid-feasibility-v1','classes':list(CLASSES)}},checkpoint)
            image=Image.new('RGB',(29,13),'brown');image.save(root/'photo.png')
            probabilities=predict_lesions(checkpoint,image)
            self.assertEqual(probabilities.shape,(4,13,29))
            for c,bias in enumerate((-2.,0.,2.,4.)):
                np.testing.assert_allclose(probabilities[c],1/(1+np.exp(-bias)),atol=1e-6)
            report=save_lesion_prediction(checkpoint,root/'photo.png',root/'out')
            self.assertFalse(report['probabilities_saved']);self.assertEqual(report['classes'],list(CLASSES))
            self.assertFalse((root/'out/probabilities.npy').exists())
            self.assertEqual(int(np.asarray(Image.open(root/'out/MA.png')).sum()),0)
            self.assertTrue((np.asarray(Image.open(root/'out/EX.png'))==255).all())
            with self.assertRaises(FileExistsError):save_lesion_prediction(checkpoint,root/'photo.png',root/'out')

    def test_binary_checkpoint_is_not_silently_loaded_as_four_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'binary.pt';torch.save({'model':{},'config':{'architecture':'query'}},path)
            with self.assertRaisesRegex(ValueError,'four-channel'):predict_lesions(path,Image.new('RGB',(29,13)))


if __name__=='__main__':unittest.main()
