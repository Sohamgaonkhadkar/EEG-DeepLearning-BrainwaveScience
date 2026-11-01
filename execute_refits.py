"""Wait for required development runs, commit the numeric plan, then execute refits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
ENV=os.environ.copy()
ENV.update(PYTHONIOENCODING='utf-8',MNE_DATASETS_EEGBCI_PATH=str(ROOT.parent/'mne_data'),
           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
expected={(m,s) for m in ['EEGNet','VanillaEEGTransformer','YourModel'] for s in [0,1,2]}
while True:
    file=ROOT/'part2_transformer/results.jsonl'
    try:
        rows=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
        complete={(r['model'],r['seed']) for r in rows if r['epochs']==40}
        if expected <= complete and (ROOT/'part2_transformer/figure.png').exists(): break
    except (FileNotFoundError,json.JSONDecodeError): pass
    time.sleep(5)
print('Development runs complete; freezing the validation-only refit plan',flush=True)
plan=subprocess.check_output([sys.executable,'refit.py','--prepare-plan'],cwd=ROOT,env=ENV,text=True,encoding='utf-8')
(ROOT/'logs/refit_plan.log').write_text(plan,encoding='utf-8')
print(plan,flush=True)
subprocess.run(['git','add','refit_plan.json'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Freeze validation-derived durations before eight-subject refitting'],cwd=ROOT,check=True)
with (ROOT/'logs/refits.log').open('w',encoding='utf-8') as log:
    log.write('Command: '+sys.executable+' refit.py\n');log.flush()
    process=subprocess.Popen([sys.executable,'-u','refit.py'],cwd=ROOT,env=ENV,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace')
    for line in process.stdout:
        log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=process.wait()
if code: raise RuntimeError(f'Refit failed: {code}; see logs/refits.log')
print('REFIT EXECUTION COMPLETE',flush=True)
