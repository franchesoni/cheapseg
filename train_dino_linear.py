print("importing standard...")
from pathlib import Path

print("importing external...")
import torch
import numpy as np
from fire import Fire
from PIL import Image
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_HSV2RGB


print("done importing.")


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

    def __init__(self, ds_path, split="train"):
        ds_path = Path(ds_path)
        split = "validation" if split in ["validation", "test", "val"] else "training"
        img_dir = ds_path / "images" / split
        ann_dir = ds_path / "annotations" / split
        img_files = sorted(img_dir.glob("*.jpg"))
        ann_files = sorted(ann_dir.glob("*.png"))
        assert len(img_files) == len(
            ann_files
        ), "Mismatched number of images and annotations"
        assert (img_files[0].stem == ann_files[0].stem) and (
            img_files[-1].stem == ann_files[-1].stem
        ), "Mismatched files"
        self.samples = list(zip(img_files, ann_files))
        self.crop_size = 512

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        # here we must map the background to 255
        original_shape = img.height, img.width
        zoom_factor = np.random.uniform(0.5, 2)
        img_shape = (
            int(original_shape[0] * zoom_factor),
            int(original_shape[1] * zoom_factor),
        )
        img = img.resize((img_shape[1], img_shape[0]), Image.BILINEAR)
        ann = ann.resize((img_shape[1], img_shape[0]), Image.NEAREST)
        if self.crop_size < img_shape[0] or self.crop_size < img_shape[1]:
            for _ in range(10):
                margin_h, margin_w = (
                    max(-self.crop_size + img_shape[0], 0),
                    max(-self.crop_size + img_shape[1], 0),
                )
                crop_origin = np.random.randint(0, margin_h + 1), np.random.randint(
                    0, margin_w + 1
                )
                ann_crop = ann.crop(
                    (
                        crop_origin[1],
                        crop_origin[0],
                        min(crop_origin[1] + self.crop_size, img_shape[1]),
                        min(crop_origin[0] + self.crop_size, img_shape[0]),
                    )
                )
                hist = ann_crop.histogram()[1:-1]  # both 0 and 255 are background
                if max(hist) / sum(hist) < 0.75:
                    ann = ann_crop
                    break
            img = img.crop(
                (
                    crop_origin[1],
                    crop_origin[0],
                    min(crop_origin[1] + self.crop_size, img_shape[1]),
                    min(crop_origin[0] + self.crop_size, img_shape[0]),
                )
            )
        # random flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann.transpose(Image.FLIP_LEFT_RIGHT)
        # go to np
        img = np.array(img).astype(np.float32)
        ann = np.array(ann)
        # handle background as in mmseg
        ann[ann == 0] = 255
        ann = ann - 1
        ann[ann == 254] = 255
        # sample photometric distortion
        change_brightness = np.random.randint(0, 2)
        change_contrast = np.random.randint(0, 2)
        contrast_last = np.random.randint(0, 2)
        change_saturation = np.random.randint(0, 2)
        change_hue = np.random.randint(0, 2)
        if change_brightness:
            img = np.clip(img + np.random.randint(-32, 32), 0, 255)
        if change_contrast and not contrast_last:
            alpha = np.random.uniform(0.5, 1.5)
            img = np.clip(img * alpha, 0, 255)
        if change_saturation or change_hue:
            hsv = cvtColor(img.astype(np.uint8), COLOR_RGB2HSV)
            if change_saturation:
                alpha = np.random.uniform(0.5, 1.5)
                hsv[..., 1] = np.clip(hsv[..., 1] * alpha, 0, 255)
            if change_hue:
                alpha = np.random.uniform(-18, 18)
                hsv[..., 0] = (hsv[..., 0] + alpha) % 180
            img = cvtColor(hsv, COLOR_HSV2RGB)
        if change_contrast and contrast_last:
            alpha = np.random.uniform(0.5, 1.5)
            img = np.clip(img * alpha, 0, 255)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, ann


def main(
    ds_path, model_name="dinov2_vitb14_reg", batch_size=16, num_workers=10, seed=0
):
    ds = ADE20K(ds_path)
    np.random.seed(seed)
    for _ in range(10):
        img, ann = ds[0]
        Image.fromarray(img).save("aimg.png")
        Image.fromarray(ann).save("aann.png")
        breakpoint()


if __name__ == "__main__":
    Fire(main)
