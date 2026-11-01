"""Download the original and corrected task runs with MNE checksum verification."""
import concurrent.futures
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MNE_DATASETS_EEGBCI_PATH']=str(ROOT.parent/'mne_data')
from mne.datasets import eegbci

def fetch(s):
 paths=eegbci.load_data(s,[5,6,9,10,13,14],update_path=False,verbose='ERROR')
 print(f'Subject {s}: {len(paths)} recordings cached',flush=True)
 return paths

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for paths in pool.map(fetch, range(1,11)): pass

