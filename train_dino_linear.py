
import torch
from fire import Fire
from pathlib import Path

class ADE20K(torch.utils.data.Dataset):
    """
    Expected directory structure:

    │   ├── ade
    │   │   ├── ADEChallengeData2016 <- this is ds_path
    │   │   │   ├── annotations
    │   │   │   │   ├── training
    │   │   │   │   ├── validation
    │   │   │   ├── images
    │   │   │   │   ├── training
    │   │   │   │   ├── validation
    """
    def __init__(self, ds_path, split='train'):
        ds_path = Path(ds_path)
        split = 'validation' if split in ['validation', 'test', 'val'] else 'training'
        img_dir = ds_path / 'images' / split 
        ann_dir = ds_path / 'annotations' / split
        img_files = sorted(img_dir.glob('*.jpg'))
        ann_files = sorted(ann_dir.glob('*.png'))
        assert len(img_files) == len(ann_files), "Mismatched number of images and annotations"
        assert (img_files[0].stem == ann_files[0].stem) and (img_files[-1].stem == ann_files[-1].stem), "Mismatched files"

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def main(
        ds_path,
        model_name='dinov2_vitb14_reg',
        batch_size=16,
        num_workers=10,
):
    ds = ADE20K(ds_path)

if __name__ == '__main__':
    Fire(main)