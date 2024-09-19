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

    # increase font size
    plt.rc('font', size=18)
    plt.figure()
    name_map = {"knet": "K-Net",
                "dinov2_vits14_": "DINOv2-small",
                "dinov2_vitb14_": "DINOv2-base",
                "dinov2_vitb14reg": "DINOv2reg-base"}

    for model_idx, model_name in enumerate(['knet', 'dinov2_vits14_', 'dinov2_vitb14_', 'dinov2_vitb14reg']):
        model_entries = {k: v for k, v in entries.items() if model_name in k}
        ds_entries = {(int(k.split('ds')[1].split('-')[0].split('_')[0]) if 'ds' in k else 1): v for k, v in model_entries.items()}
        sorted_entries = dict(sorted(ds_entries.items(), key=lambda item: item[0], reverse=False))
        # plt.plot(np.log2(list(sorted_entries.keys())), sorted_entries.values(), '-o', label=name_map[model_name])
        if model_name == 'dinov2_vitb14_':
            # remove last entry
            sorted_entries.pop(1)
        linestyle = ["-", "--", ":", "-."][model_idx]
        plt.plot(20210 / np.array(list(sorted_entries.keys())), sorted_entries.values(), linestyle, label=name_map[model_name], color='k')
        plt.plot(20210 / np.array(list(sorted_entries.keys())), sorted_entries.values(), '.', color='k')

    plt.ylabel('ADE20k val mIoU')
    # plt.xlabel('$\log_2$(dataset downsampling factor)')
    plt.xscale('log')
    plt.grid()
    plt.xlabel('Number of annotated images')#$\log_2$(dataset downsampling factor)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ious2.png')

if __name__ == '__main__':
    main()


        

