"""Freeze validation-derived durations, then refit on all eight development people.

Never reads a development test score. All refits finish before test data loading.
The required development measurements remain unchanged in each part directory.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MNE_DATASETS_EEGBCI_PATH',str(ROOT.parent/'mne_data'))
sys.path.insert(0,str(ROOT/'part2_transformer'))
from main import YourModel
from models import EEGNet, VanillaEEGTransformer, count_params
from train import set_seed, evaluate
from data import _load_subjects, RUNS
from eeg_quality import standardize_training, validate_subject, save_audit
from mne.datasets import eegbci
from mne.io import read_raw_edf


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1DATA=load_module('refit_p1_data',ROOT/'part1_diagnose/data.py')
P1MODEL=load_module('refit_p1_models',ROOT/'part1_diagnose/model.py')


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare_plan():
    """Only validation histories and declared hyperparameters determine the plan."""
    p1=jsonl(ROOT/'part1_diagnose/results.jsonl')
    p2=jsonl(ROOT/'part2_transformer/results.jsonl')
    plan=dict(rule='per-model median of earliest validation-maximizing epochs across seeds 0/1/2; Part 1 uses its single seed',
              train_subjects=list(range(1,9)),test_subjects=[9,10],
              learning_rate=1e-3,weight_decay=1e-3,
              test_status='Previously scored in required development comparisons; no test metric used in refit decisions',
              part1={},part2={})
    for channels in ['all','frontal']:
        record=next(r for r in reversed(p1) if r['channels']==channels and r['seed']==0)
        history=record['history']
        best=max(h['validation_acc'] for h in history)
        epoch=next(h['epoch']+1 for h in history if h['validation_acc']==best)
        plan['part1'][channels]=dict(epochs=epoch,seeds=[0],batch_size=64,
                                    channels=record['selected_channels'])
    for name in ['EEGNet','VanillaEEGTransformer','YourModel']:
        epochs=[]
        for seed in [0,1,2]:
            record=next(r for r in reversed(p2) if r['model']==name and r['seed']==seed and r['epochs']==40)
            history=record['history']
            best=max(h['validation_acc'] for h in history)
            epochs.append(next(h['epoch']+1 for h in history if h['validation_acc']==best))
        plan['part2'][name]=dict(epochs=int(np.median(epochs)),
                                development_best_epochs=epochs,seeds=[0,1,2],batch_size=16)
    path=ROOT/'refit_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError('Existing frozen refit plan differs; refusing silent overwrite')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)


def fixed_train(model,X,y,epochs,batch,device):
    """No validation/test arrays or scores are accepted; use final fixed epoch."""
    model.to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),
                      batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    history=[]
    for epoch in range(epochs):
        model.train(); loss_sum=0.0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(xb),yb)
            loss.backward(); opt.step()
            loss_sum += loss.item()*len(yb)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(y)))
        print(f'  fixed epoch {epoch+1}/{epochs} train_loss {loss_sum/len(y):.4f}',flush=True)
    return history


def load_p2(subjects,audit_path):
    Xs,ys,groups,audits=[],[],[],[]
    for subject in subjects:
        X,y=_load_subjects([subject])
        files=eegbci.load_data(subject,RUNS[:1],update_path=False,verbose='ERROR')
        header=read_raw_edf(files[0],preload=False,verbose='ERROR')
        eegbci.standardize(header)
        audits.append(validate_subject(X,y,header.ch_names,subject,
                                       header.info['sfreq'],RUNS,'V',np.array([0,3])))
        Xs.append(X); ys.append(y); groups.extend([subject]*len(y))
    save_audit(audits,audit_path)
    return np.concatenate(Xs),np.concatenate(ys),np.array(groups)


def run_refits():
    plan=json.loads((ROOT/'refit_plan.json').read_text())
    if plan['train_subjects'] != list(range(1,9)) or plan['test_subjects'] != [9,10]:
        raise ValueError('Invalid refit subjects')
    destination=ROOT/'refits'
    destination.mkdir(exist_ok=True)
    for part in ['part1_diagnose','part2_transformer']:
        (destination/part).mkdir(exist_ok=True)
        if (destination/part/'results.jsonl').exists():
            raise RuntimeError('Refit result file exists; refusing duplicate experimental records')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    # Load and fit preprocessing using only development people.
    os.chdir(destination/'part1_diagnose')
    raw1,y1,g1,ch=P1DATA.build_dataset(list(range(1,9)))
    Path('data_audit.json').rename('train_data_audit.json')
    X1=standardize_training(raw1)[0]
    raw2,y2,g2=load_p2(list(range(1,9)),destination/'part2_transformer/train_data_audit.json')
    X2=standardize_training(raw2)[0]
    fitted=[]
    for channels,configuration in plan['part1'].items():
        picks=[ch.index(c) for c in configuration['channels']]
        set_seed(0)
        model=P1MODEL.EEGNet(len(picks),X1.shape[2])
        print(f'REFIT Part 1 {channels}: 336 development trials, epochs={configuration["epochs"]}',flush=True)
        start=time.perf_counter()
        history=fixed_train(model,X1[:,picks],y1,configuration['epochs'],configuration['batch_size'],device)
        fitted.append(('part1_diagnose',channels,0,model,picks,history,time.perf_counter()-start))
    C,T=X2.shape[1:]
    builders={'EEGNet':lambda: EEGNet(C,T,kern=80,p=.5),
              'VanillaEEGTransformer':lambda: VanillaEEGTransformer(C,T),
              'YourModel':lambda: YourModel(C,T)}
    for name,configuration in plan['part2'].items():
        for seed in configuration['seeds']:
            set_seed(seed); model=builders[name]()
            print(f'REFIT Part 2 {name} seed {seed}: 360 development trials, epochs={configuration["epochs"]}',flush=True)
            start=time.perf_counter()
            history=fixed_train(model,X2,y2,configuration['epochs'],configuration['batch_size'],device)
            fitted.append(('part2_transformer',name,seed,model,None,history,time.perf_counter()-start))
    print('ALL REFIT WEIGHTS FROZEN; loading test people for final scoring',flush=True)
    os.chdir(destination/'part1_diagnose')
    raw1te,y1te,g1te,ch_te=P1DATA.build_dataset([9,10])
    Path('data_audit.json').rename('test_data_audit.json')
    assert ch==ch_te
    X1te=standardize_training(raw1,raw1te)[1]
    raw2te,y2te,g2te=load_p2([9,10],destination/'part2_transformer/test_data_audit.json')
    X2te=standardize_training(raw2,raw2te)[1]
    for part,name,seed,model,picks,history,seconds in fitted:
        Xte,yte,groups=(X1te[:,picks],y1te,g1te) if part=='part1_diagnose' else (X2te,y2te,g2te)
        loader=DataLoader(TensorDataset(torch.from_numpy(Xte),torch.from_numpy(yte)),batch_size=16)
        accuracy,loss,predictions,labels=evaluate(model,loader,device,True)
        predictions=np.array(predictions)
        record=dict(model=name,seed=seed,protocol='all_eight_subject_refit',
                    epochs=len(history),checkpoint='predetermined_final_epoch',
                    duration_rule=plan['rule'],n_train=len(y1) if part=='part1_diagnose' else len(y2),
                    n_test=len(yte),train_subjects=list(range(1,9)),test_subjects=[9,10],
                    normalization_fit_subjects=list(range(1,9)),n_params=count_params(model),
                    lr=1e-3,weight_decay=1e-3,batch_size=64 if part=='part1_diagnose' else 16,
                    test_acc=accuracy,test_loss=loss,history=history,seconds=seconds,
                    test_predictions=predictions.tolist(),test_labels=labels,
                    confusion_matrix=confusion_matrix(yte,predictions,labels=[0,1]).tolist(),
                    per_class=classification_report(yte,predictions,labels=[0,1],target_names=['fists','feet'],output_dict=True,zero_division=0),
                    per_subject_accuracy={str(s):float(np.mean(predictions[groups==s]==yte[groups==s])) for s in [9,10]})
        with (destination/part/'results.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(f'REFIT RESULT {part} {name} seed {seed}: {accuracy:.4f}',flush=True)
    print('ALL REFITS COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-plan',action='store_true')
    args=parser.parse_args()
    if args.prepare_plan: prepare_plan()
    else: run_refits()
