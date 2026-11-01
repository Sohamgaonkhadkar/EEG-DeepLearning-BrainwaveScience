"""Resume only missing mandatory runs, preserving completed records."""
from run_assessment import run, ROOT
if __name__ == '__main__':
    run('yourmodel_original_3seeds', ROOT/'part2_transformer',
        ['main.py','--models','YourModel','--epochs','40','--seeds','0','1','2'])
    run('vanilla_seed2_restart', ROOT/'part2_transformer',
        ['main.py','--models','VanillaEEGTransformer','--epochs','40','--seeds','2'])
