"""Plot preselected seed-0 training histories after all development runs finish."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (ROOT/'part2_transformer/results.jsonl').read_text().splitlines() if s.strip()]
fig,axes=plt.subplots(1,3,figsize=(13,3.7))
summary={}
for ax,name in zip(axes,['EEGNet','VanillaEEGTransformer','YourModel']):
    records=[r for r in rows if r['model']==name and r['epochs']==40]
    assert records
    record=next(r for r in records if r['seed']==0)
    h=record['history']; x=np.arange(1,len(h)+1)
    ax.plot(x,[p['train_loss'] for p in h],label='training (online)',color='tab:blue')
    if 'train_eval_loss' in h[0]:
        ax.plot(x,[p['train_eval_loss'] for p in h],label='training (eval)',color='tab:green')
    ax.plot(x,[p['validation_loss'] for p in h],label='validation',color='tab:orange')
    ax.axvline(record['selected_epoch']+1,color='gray',ls='--',lw=1,label='selected epoch')
    ax.set(title={'YourModel':'Temporal-Patch EEG\nTransformer','VanillaEEGTransformer':'Vanilla EEG Transformer'}.get(name,name),xlabel='epoch',ylabel='cross entropy')
    ax.set_yscale('log'); ax.grid(alpha=.2)
    summary[name]=dict(final_training_loss_mean=float(np.mean([r['history'][-1]['train_loss'] for r in records])),
                       final_validation_loss_mean=float(np.mean([r['history'][-1]['validation_loss'] for r in records])),
                       mean_training_seconds=float(np.mean([r['seconds'] for r in records])))
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle('Development loss curves: seed 0 selected in advance (logarithmic loss axis)')
fig.tight_layout();fig.savefig(ROOT/'part2_transformer/training_curves.png',dpi=150)
(ROOT/'part2_transformer/training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
