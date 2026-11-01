"""Execute the frozen assessment configuration and preserve complete transcripts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ENV = os.environ.copy()
ENV.update(PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
           MNE_DATASETS_EEGBCI_PATH=str(WORKSPACE/'mne_data'))
LOGS = ROOT/'logs'
LOGS.mkdir(exist_ok=True)


def run(label, directory, args):
    print(f'START {label}',flush=True)
    with (LOGS/f'{label}.log').open('w',encoding='utf-8') as log:
        log.write('Command: '+str(sys.executable)+' '+ ' '.join(args)+'\n')
        log.write('Working directory: '+str(directory)+'\n')
        log.flush()
        proc=subprocess.Popen([sys.executable,'-u',*args],cwd=directory,env=ENV,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace')
        # The untouched starter asks once per subject to persist its cache path.
        # Decline without editing starter source or the user's MNE config.
        proc.stdin.write("n\n" * 20)
        proc.stdin.close()
        for line in proc.stdout:
            log.write(line); log.flush()
            if len(line)<300: print(label+': '+line.rstrip(),flush=True)
        code=proc.wait()
    if code: raise RuntimeError(f'{label} failed ({code}); see logs/{label}.log')
    print(f'COMPLETE {label}',flush=True)


if __name__=='__main__':
    if '--wait-data' in sys.argv:
        paths=[WORKSPACE/'mne_data'/'MNE-eegbci-data'/'files'/'eegmmidb'/'1.0.0'/
               f'S{s:03d}'/f'S{s:03d}R{r:02d}.edf'
               for s in range(1,11) for r in [5,6,9,10,13,14]]
        while not all(p.exists() for p in paths): time.sleep(5)
    original=WORKSPACE/'original_starter'/'eeg_dl_exam_3hr'/'part1_diagnose'
    jobs=[('original_part1',original,['train.py','--channels','all','--seed','0']),
          ('corrected_all',ROOT/'part1_diagnose',['train.py','--channels','all','--seed','0']),
          ('corrected_frontal',ROOT/'part1_diagnose',['train.py','--channels','frontal','--seed','0']),
          ('erd_mu',ROOT/'part1_diagnose',['erd.py','--band','mu']),
          ('erd_beta',ROOT/'part1_diagnose',['erd.py','--band','beta']),
          ('part2_40epochs_3seeds',ROOT/'part2_transformer',
           ['main.py','--epochs','40','--seeds','0','1','2'])]
    resume=sys.argv[sys.argv.index('--resume')+1] if '--resume' in sys.argv else jobs[0][0]
    labels=[job[0] for job in jobs]
    if resume not in labels: raise ValueError(f'Unknown resume stage: {resume}')
    for label,directory,args in jobs[labels.index(resume):]:
        run(label,directory,args)
        if label=='original_part1':
            shutil.copy2(original/'results.jsonl',ROOT/'part1_diagnose'/'original_results.jsonl')
    print('ALL EXPERIMENTS COMPLETE',flush=True)
