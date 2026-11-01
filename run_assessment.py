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
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace')
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
    run('original_part1',original,['train.py','--channels','all','--seed','0'])
    shutil.copy2(original/'results.jsonl',ROOT/'part1_diagnose'/'original_results.jsonl')
    run('corrected_all',ROOT/'part1_diagnose',['train.py','--channels','all','--seed','0'])
    run('corrected_frontal',ROOT/'part1_diagnose',['train.py','--channels','frontal','--seed','0'])
    run('erd_mu',ROOT/'part1_diagnose',['erd.py','--band','mu'])
    run('erd_beta',ROOT/'part1_diagnose',['erd.py','--band','beta'])
    run('part2_40epochs_3seeds',ROOT/'part2_transformer',
        ['main.py','--epochs','40','--seeds','0','1','2'])
    print('ALL EXPERIMENTS COMPLETE',flush=True)
