import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main(work_dirs_dir='work_dirs'):
    work_dirs_dir = Path(work_dirs_dir)
    entries = {}
    for work_dir in os.listdir(work_dirs_dir):
        if not (work_dirs_dir / work_dir).is_dir():
            continue
        last_run_dir = work_dirs_dir / work_dir / (sorted([p for p in os.listdir(work_dirs_dir / work_dir) if (work_dirs_dir / work_dir / p).is_dir()])[-1])
        scalars_filepath = last_run_dir / 'vis_data' / 'scalars.json'
        if not scalars_filepath.exists():
            continue
        with open(scalars_filepath, 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        val_lines = [line['mIoU'] for line in lines if 'mIoU' in line]
        if len(val_lines):
            entries[work_dir] = max(val_lines)

    plt.figure()

    knet_entries = {k: v for k, v in entries.items() if 'knet' in k}
    ds_entries = {(int(k.split('ds')[1].split('-')[0]) if 'ds' in k else 1): v for k, v in knet_entries.items()}
    sorted_entries = dict(sorted(ds_entries.items(), key=lambda item: item[0], reverse=True))
    plt.plot(np.log2(list(sorted_entries.keys())), sorted_entries.values(), '-o', label='knet')

    dinov2_entries = {k: v for k, v in entries.items() if 'dinov2' in k}
    ds_entries = {(int(k.split('ds')[1].split('_')[0]) if 'ds' in k else 1): v for k, v in dinov2_entries.items()}
    sorted_entries = dict(sorted(ds_entries.items(), key=lambda item: item[0], reverse=True))
    plt.plot(np.log2(list(sorted_entries.keys())), sorted_entries.values(), '-o', label='dinov2')

    plt.ylabel('mIoU')
    plt.xlabel('$\log_2$(dataset downsampling factor)')
    plt.legend()
    plt.savefig('ious.png')

if __name__ == '__main__':
    main()


        

