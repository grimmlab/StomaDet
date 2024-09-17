import subprocess
from sklearn.model_selection import ParameterGrid
from scipy.stats import loguniform
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    out_folder = Path("./output").glob('**/*')
    all_files = [str(x).split("/")[-1] for x in out_folder if x.is_dir()]
    seed = 42
    max_iter = 15000
    lower_lr = 5e-4
    upper_lr = 5e-3
    batch_sizes = [1,2,3,4,8,16,32]
    rois = [128, 256, 512]
    np.random.seed(seed)
    learning_rates = loguniform.rvs(lower_lr, upper_lr, size=10)
    param_grid = {'lr': learning_rates, 'bs': batch_sizes, 'roi': rois}
    parameters = list(ParameterGrid(param_grid))
    print(all_files)
    for idx, val in enumerate(parameters):

        print(f"starting run {idx}/{len(parameters)}")
        folder_str = f"{val['lr']}_{val['bs']}_{val['roi']}_{max_iter}"
        print(folder_str)
        if folder_str in all_files:
            print(f"skipping run for {folder_str}")
        else:
            print(f"Training run {idx}: {val}")
            subprocess.run(f"python3 train_faster_rcnn.py --lr {val['lr']} --bs {val['bs']} --roi {val['roi']} --max_iter {max_iter}", shell=True, check=True)
