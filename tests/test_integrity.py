"""Meaningful invariant tests; synthetic tensors are never experimental results."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part2_transformer"))
from main import YourModel
from models import EEGNet, VanillaEEGTransformer
from train import train_model, set_seed
sys.path.insert(0, str(ROOT))
from eeg_quality import validate_subject, save_audit


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        set_seed(0)

    def test_forward_backward_all_models(self):
        for build in [EEGNet, VanillaEEGTransformer, YourModel]:
            model = build(64, 481)
            x = torch.randn(2, 64, 481)
            out = model(x)
            self.assertEqual(out.shape, (2, 2))
            torch.nn.functional.cross_entropy(out, torch.tensor([0,1])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))

    def test_patches_cover_last_sample(self):
        model = YourModel(64, 481).eval()
        self.assertEqual(model.pos.shape, (1, 17, 64))
        x = torch.zeros(1,64,481, requires_grad=True)
        model(x).sum().backward()
        self.assertGreater(x.grad[:,:,-1].abs().sum().item(),0)
        logits, embed = model(x, return_embed=True)
        self.assertEqual(embed.shape, (1,64))

    def test_validation_checkpoint_matches_history(self):
        x=np.random.randn(8,2,20).astype("float32")
        y=np.array([0,1]*4,dtype="int64")
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(40,2))
        best,h=train_model(model,x[:4],y[:4],x[4:],y[4:],epochs=3,batch=2)
        self.assertEqual(best,max(r["validation_acc"] for r in h))
        self.assertEqual(model.selected_epoch,next(r["epoch"] for r in h if r["validation_acc"]==best))
        self.assertTrue(all("test_acc" not in r for r in h))

    def test_erd_reference_and_sign(self):
        # Load Part 1 without shadowing the Part 2 data module.
        old=sys.modules.pop("data",None)
        sys.path.insert(0,str(ROOT/"part1_diagnose"))
        spec=importlib.util.spec_from_file_location("erd_check",ROOT/"part1_diagnose/erd.py")
        erd=importlib.util.module_from_spec(spec); spec.loader.exec_module(erd)
        sys.path.pop(0)
        sys.modules.pop("data",None)
        if old is not None: sys.modules["data"]=old
        class Epochs:
            times=np.array([-1.5,-1.0,-0.5,0.0,1.0])
            def __getitem__(self,key): return self
            def get_data(self,copy=True): return np.array([[[4.,4.,4.,2.,2.]]])
        values,times=erd.erd_percent(Epochs(),"hands")
        np.testing.assert_allclose(values[0,:3],0)
        np.testing.assert_allclose(values[0,3:],-50)

    def test_invalid_data_rejected(self):
        channels=["C3","Cz","C4"]+[f"X{i}" for i in range(61)]
        x=np.random.randn(2,64,20).astype("float32"); y=np.array([0,1])
        x[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"non-finite"):
            validate_subject(x,y,channels,1,160,[6,10,14],"uV",[0,3])


if __name__=="__main__": unittest.main()
