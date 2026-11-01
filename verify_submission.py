"""Verify submitted measurements, data hashes, split boundaries and artifacts."""
from pathlib import Path
import hashlib
import importlib.resources
import json
import subprocess
import zipfile
import sys
import numpy as np
import mne
from PIL import Image
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent


def records(path):
 return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
 checks=[]
 for commit in ['11f4b6f','8c1dca8','c4ede07','24c4e0d','ded795d','3fe8efd']:
  subprocess.check_call(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT)
 checks.append('All six separate scientific bug-fix commits remain in local history')
 registry={line.split()[0]:line.split()[1] for line in
           importlib.resources.files('mne').joinpath('data/eegbci_checksums.txt').read_text().splitlines() if line.strip()}
 for subject in range(1,11):
  for run in [5,6,9,10,13,14]:
   key=f'S{subject:03d}/S{subject:03d}R{run:02d}.edf'
   file=ROOT.parent/'mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0'/key
   assert hashlib.sha256(file.read_bytes()).hexdigest()==registry[key],key
 checks.append('All 60 source EDF SHA256 hashes match MNE registry')
 for part in ['part1_diagnose','part2_transformer']:
  audit=json.loads((ROOT/part/'data_audit.json').read_text())
  assert [a['subject'] for a in audit]==list(range(1,11))
  assert all(a['sfreq']==160 and a['runs']==[6,10,14] and a['finite'] for a in audit)
  hashes=[set(a['epoch_hashes']) for a in audit]
  assert all(not hashes[i]&hashes[j] for i in range(10) for j in range(i))
  assert all(a['channels']==audit[0]['channels'] for a in audit)
  for r in records(ROOT/part/'results.jsonl'):
   assert r['train_subjects']==list(range(1,7))
   assert r['validation_subjects']==[7,8] and r['test_subjects']==[9,10]
   pred=np.array(r['test_predictions']); y=np.array(r['test_labels'])
   assert len(y)==len(pred)==r['n_test']
   assert abs(np.mean(pred==y)-r['test_acc'])<1e-12
   assert confusion_matrix(y,pred,labels=[0,1]).tolist()==r['confusion_matrix']
   history=r['history']; assert len(history)==r['epochs']
   selected=max(h['validation_acc'] for h in history)
   ep=next(h['epoch'] for h in history if h['validation_acc']==selected)
   assert r.get('best_epoch',r.get('selected_epoch'))==ep
   assert selected==r['best_validation_acc']
 checks.append('Actual result accuracies/confusion matrices and validation checkpoint histories recompute exactly')
 p2=records(ROOT/'part2_transformer/results.jsonl')
 expected={(m,s) for m in ['EEGNet','VanillaEEGTransformer','YourModel'] for s in [0,1,2]}
 actual={(r['model'],r['seed']) for r in p2 if r['epochs']==40}
 missing=sorted(expected-actual)
 assert len(p2)==len(actual), 'Duplicate or non-equivalent completed results'
 assert not actual-expected
 assert not missing or ('--allow-incomplete' in sys.argv and set(missing)<={('VanillaEEGTransformer',2)})
 checks.append('Completed seed records verified; missing runs: '+str(missing))
 for r in p2:
  if 'checkpoint_file' in r:
   assert (ROOT/'part2_transformer'/r['checkpoint_file']).exists()
   assert r['checkpoint_reload_verified']
   assert all('train_acc' in h and 'train_eval_loss' in h for h in r['history'])
 historical=records(ROOT/'audit_snapshot_before_stop/part2_transformer/results.jsonl')
 assert p2[:len(historical)]==historical
 assert (ROOT/'part2_transformer/results.jsonl').read_bytes().startswith((ROOT/'audit_snapshot_before_stop/part2_transformer/results.jsonl').read_bytes())
 for name in ['results.jsonl','original_results.jsonl']:
  assert (ROOT/'part1_diagnose'/name).read_bytes()==(ROOT/'audit_snapshot_before_stop/part1_diagnose'/name).read_bytes()
 checks.append('Historical five runs unchanged; new checkpoints and training metrics present')
 if (ROOT/'refits/part2_transformer/results.jsonl').exists():
  plan=json.loads((ROOT/'refit_plan.json').read_text())
  for part,n_train,n_test in [('part1_diagnose',336,84),('part2_transformer',360,90)]:
   refits=records(ROOT/'refits'/part/'results.jsonl')
   assert len(refits)==(2 if part=='part1_diagnose' else 9)
   for r in refits:
    config=plan['part1' if part=='part1_diagnose' else 'part2'][r['model']]
    assert r['epochs']==config['epochs']==len(r['history'])
    assert r['n_train']==n_train and r['n_test']==n_test
    assert r['train_subjects']==r['normalization_fit_subjects']==list(range(1,9))
    assert r['test_subjects']==[9,10] and r['checkpoint']=='predetermined_final_epoch'
    assert all(set(h)=={'epoch','train_loss'} for h in r['history'])
    assert abs(np.mean(np.array(r['test_predictions'])==r['test_labels'])-r['test_acc'])<1e-12
    assert confusion_matrix(r['test_labels'],r['test_predictions'],labels=[0,1]).tolist()==r['confusion_matrix']
   training=json.loads((ROOT/'refits'/part/'train_data_audit.json').read_text())
   testing=json.loads((ROOT/'refits'/part/'test_data_audit.json').read_text())
   assert [a['subject'] for a in training]==list(range(1,9))
   assert [a['subject'] for a in testing]==[9,10]
   train_hashes={h for a in training for h in a['epoch_hashes']}
   test_hashes={h for a in testing for h in a['epoch_hashes']}
   assert not train_hashes & test_hashes
  refit_log=(ROOT/'logs/refits.log').read_text()
  assert refit_log.index('ALL REFIT WEIGHTS FROZEN') < refit_log.index('REFIT RESULT')
  checks.append('All eleven eight-subject refits follow frozen durations, train-only scaling and full test sets')
 else:
  checks.append('Optional refits not executed; excluded from mandatory completion')

 for file in ['part1_diagnose/erd_mu.png','part1_diagnose/erd_beta.png','part2_transformer/figure.png']:
  with Image.open(ROOT/file) as image:
   image.verify()
 checks.append('All required figures and supplementary beta ERD are valid PNGs')
 for file in ['ANSWERS.md','WALKTHROUGH_NOTES.md','part1_diagnose/original_results.jsonl']:
  assert (ROOT/file).stat().st_size>0
 for file in ['part1_diagnose/model.py','part2_transformer/models.py']:
  before=subprocess.check_output(['git','show',f'b76b586:{file}'],cwd=ROOT)
  assert before.replace(b'\r\n',b'\n')==(ROOT/file).read_bytes().replace(b'\r\n',b'\n')
 checks.append('Part 1 EEGNet architecture and Part 2 baseline model source are unchanged')
 with zipfile.ZipFile(ROOT.parent/'eeg_dl_exam_3hr (1).zip') as z:
  for name in z.namelist():
   if not name.endswith('/'):
    original=ROOT.parent/'original_starter'/name
    assert original.read_bytes()==z.read(name),name
 checks.append('Original starter remains byte-identical to supplied ZIP')
 assert not subprocess.check_output(['git','remote'],cwd=ROOT).strip()
 tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
 assert not any(p.endswith(('.edf','.pt','.pth')) or '.venv/' in p for p in tracked)
 checks.append('No Git remote configured; raw datasets and checkpoints not tracked')
 (ROOT/'verification.json').write_text(json.dumps(dict(passed=True,comparison_complete=not missing,missing_runs=missing,checks=checks),indent=2)+'\n')
 print('\n'.join(checks))

if __name__=='__main__': main()
