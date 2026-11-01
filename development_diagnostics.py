"""Development-only amplitude and saved-validation diagnostics; never load test data."""
from pathlib import Path
import os, sys, json
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
ROOT=Path(__file__).resolve().parent
os.environ['MNE_DATASETS_EEGBCI_PATH']=str(ROOT.parent/'mne_data')
sys.path.insert(0,str(ROOT/'part2_transformer'))
from data import _load_subjects

def prediction_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    cm=confusion_matrix(y,p,labels=[0,1])
    return dict(accuracy=float(np.mean(y==p)), balanced_accuracy=float(balanced_accuracy_score(y,p)),
        class_recall=(cm.diagonal()/cm.sum(axis=1)).tolist(),
        confusion_matrix=cm.tolist(), prediction_counts=np.bincount(p,minlength=2).tolist())

def main():
    output=ROOT/'development_diagnostics.json'
    # Cache only fixed development-data diagnostics, never test-dependent statistics.
    report=json.loads(output.read_text()) if output.exists() else {}
    if 'amplitudes' not in report:
        xs=[_load_subjects([s])[0] for s in range(1,9)]
        train=np.concatenate(xs[:6])
        mu=train.mean(axis=(0,2),keepdims=True)
        sd=train.std(axis=(0,2),keepdims=True)
        energy=np.stack([np.sum((x-mu).astype('float64')**2,axis=(0,2)) for x in xs[:6]])
        shares=energy/energy.sum(axis=0)
        flat=np.transpose(train,(1,0,2)).reshape(64,-1)
        q=np.percentile(flat,[25,50,75],axis=1)
        robust_sd=(q[2]-q[0])/1.349
        report['amplitudes']=dict(fit_subjects=list(range(1,7)), inspected_subjects=list(range(1,9)),
            channel_std_uv=(sd.ravel()*1e6).tolist(), channel_mean_uv=(mu.ravel()*1e6).tolist(),
            std_to_iqr_scale_percentiles=np.percentile(sd.ravel()/robust_sd,[0,50,95,100]).tolist(),
            subject_variance_contribution=[dict(subject=s+1, mean_share=float(shares[s].mean()),
                maximum_channel_share=float(shares[s].max())) for s in range(6)],
            subject_peak_uv=[dict(subject=i+1,peak=float(np.max(np.abs(x))*1e6)) for i,x in enumerate(xs)],
            interpretation='Variance contribution measures distribution heterogeneity, not artifact origin. No trials removed and no scaling changed.')
    rows=[json.loads(s) for s in (ROOT/'part2_transformer/results.jsonl').read_text().splitlines() if s.strip()]
    report['validation_runs']=[]
    for r in rows:
        out=dict(model=r['model'],seed=r['seed'],best_validation_acc=r['best_validation_acc'])
        if 'validation_predictions' in r:
            y,p=np.array(r['validation_labels']),np.array(r['validation_predictions'])
            out['overall']=prediction_metrics(y,p)
            out['subjects']={str(s):prediction_metrics(y[i*45:(i+1)*45],p[i*45:(i+1)*45]) for i,s in enumerate([7,8])}
        else:
            out['limitation']='Selected weights and validation predictions not saved; subject-level scores unavailable without unauthorized historical reruns.'
        report['validation_runs'].append(out)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
