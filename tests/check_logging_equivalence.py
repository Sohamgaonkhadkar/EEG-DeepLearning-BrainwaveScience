"""Prove added eval logging leaves the original optimization trajectory unchanged."""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'part2_transformer'))
from train import train_model, set_seed
old = {}
exec(subprocess.check_output(['git','show','e6097c6:part2_transformer/train.py'], cwd=ROOT, text=True), old)
torch.set_num_threads(2)
x = np.random.default_rng(41).normal(size=(24,4)).astype('float32')
y = np.arange(24, dtype='int64') % 2
def model():
    return torch.nn.Sequential(torch.nn.Linear(4,8), torch.nn.BatchNorm1d(8), torch.nn.Dropout(.3), torch.nn.Linear(8,2))
set_seed(2); a=model()
va, ha=old['train_model'](a,x,y,x[:8],y[:8],epochs=3,batch=8)
set_seed(2); b=model()
(ROOT/'cache').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
    vb,hb=train_model(b,x,y,x[:8],y[:8],epochs=3,batch=8,checkpoint_path=Path(tmp)/'selected.pt')
    assert b.checkpoint_reload_verified
assert va==vb
assert all(all(row[k]==new[k] for k in row) for row,new in zip(ha,hb))
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
print('PASS: optimization history and selected state bitwise equal; checkpoint reload verified')
