"""Finish submission artifacts locally; never train models or contact a Git remote."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
EXPECTED={(m,s) for m in ['EEGNet','VanillaEEGTransformer','YourModel'] for s in [0,1,2]}

def read_results():
    return [json.loads(s) for s in (ROOT/'part2_transformer/results.jsonl').read_text().splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--wait',action='store_true'); args=ap.parse_args()
    deadline=datetime.fromisoformat(json.loads((ROOT/'execution_deadline.json').read_text())['deadline_utc'])
    while True:
        rows=read_results()
        missing=EXPECTED-{(r['model'],r['seed']) for r in rows}
        if not missing or not args.wait or datetime.now(timezone.utc)>=deadline:
            break
        time.sleep(5)
    time.sleep(2)  # Let the completed run flush its transcript and summary.
    log=(ROOT/'logs/finalization.log').open('w',encoding='utf-8')
    def run(command):
        log.write('COMMAND: '+' '.join(command)+'\n'); log.flush()
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    run([sys.executable,'generate_report.py'])
    run([sys.executable,'plot_training.py'])
    status_path=ROOT/'EXPERIMENT_STATUS.json'
    status=json.loads(status_path.read_text())
    status.update(status='complete' if not missing else 'submission_saved_with_incomplete_run',
        completed_part2_runs=len(rows),remaining_required_part2_runs=len(missing),
        missing_runs=sorted(missing),optional_refits='not_run',compact_model='not_run',
        scaling_experiments='not_run',no_github_push=True)
    status['interrupted']['replacement_completed']=('VanillaEEGTransformer',2) not in missing
    status_path.write_text(json.dumps(status,indent=2))
    manifest=[]
    for r in rows:
        if 'checkpoint_file' in r:
            path=ROOT/'part2_transformer'/r['checkpoint_file'].replace('\\','/')
            manifest.append(dict(model=r['model'],seed=r['seed'],file=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_epoch_one_based=r['selected_epoch']+1,
                checkpoint_reload_verified=r['checkpoint_reload_verified']))
    (ROOT/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'PAUSED_FOR_REVIEW.txt').write_text('Historical pause ended by the final execution directive. See EXPERIMENT_STATUS.json for final completion status. No optional experiment watcher is running.\n')
    verify=[sys.executable,'verify_submission.py']+(['--allow-incomplete'] if missing else [])
    run(verify)
    run(['git','add','--all'])
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        run(['git','commit','-m','Finalize assessment results, checkpoints manifest, answers and verification'])
    run(verify)
    log.write('FINAL VERIFICATION PASSED; NO REMOTE ACTIONS PERFORMED\n'); log.close()
    # Finalization's own transcript is evidence and is committed after closing it.
    subprocess.run(['git','add','logs/finalization.log','verification.json'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Record final submission verification'],cwd=ROOT,check=True)
    print('FINALIZED: '+str(ROOT),flush=True)
    print('Missing runs: '+str(sorted(missing)),flush=True)

if __name__=='__main__':
    main()
