print("importing standard...")
import time
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

    def __init__(self, ds_path, split="train", crop_size=518):
        ds_path = Path(ds_path)
        self.split = (
            "validation" if split in ["validation", "test", "val"] else "training"
        )
        img_dir = ds_path / "images" / self.split
        ann_dir = ds_path / "annotations" / self.split
        img_files = sorted(img_dir.glob("*.jpg"))
        ann_files = sorted(ann_dir.glob("*.png"))
        assert len(img_files) == len(
            ann_files
        ), "Mismatched number of images and annotations"
        assert (img_files[0].stem == ann_files[0].stem) and (
            img_files[-1].stem == ann_files[-1].stem
        ), "Mismatched files"
        self.samples = list(zip(img_files, ann_files))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        original_shape = img.height, img.width

        if self.split == "training":
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
                    histsum = sum(hist)
                    if histsum and (max(hist) / histsum < 0.75):
                        break
                ann = ann_crop
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
            # bottom right pad to crop size
            pad_h = self.crop_size - img.shape[0]
            pad_w = self.crop_size - img.shape[1]
            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            ann = np.pad(
                ann,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=255,
            )
        else:
            # resize to (2048, 512) whatever hits first
            scale_factor = min(2048 / original_shape[0], 512 / original_shape[1])
            img_shape = (
                int(original_shape[0] * scale_factor),
                int(original_shape[1] * scale_factor),
            )
            img = img.resize((img_shape[1], img_shape[0]), Image.BILINEAR)
            ann = ann.resize((img_shape[1], img_shape[0]), Image.NEAREST)
            # go to np
            img = np.array(img).astype(np.float32)
            ann = np.array(ann)
            # handle background as in mmseg
            ann[ann == 0] = 255
            ann = ann - 1
            ann[ann == 254] = 255
            # pad up to size divisor of 14
            pad_h = 14 - img_shape[0] % 14
            pad_w = 14 - img_shape[1] % 14
            img = np.pad(
                np.array(img),
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            ann = np.pad(
                np.array(ann),
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=255,
            )
        if (img.shape[:2] != (self.crop_size, self.crop_size)) or (
            ann.shape[:2] != (self.crop_size, self.crop_size)
        ):
            breakpoint()

        return img.astype(np.float32), ann.astype(np.int64)


class DINOLinear(torch.nn.Module):
    def __init__(self, model_name):
        super(DINOLinear, self).__init__()
        self.bbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.in_channels = 384 if "vits" in model_name else 768
        self.bn = torch.nn.SyncBatchNorm(self.in_channels)
        self.conv_seg = torch.nn.Conv2d(self.in_channels, 150, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            out = self.bbone.get_intermediate_layers(x, n=4, reshape=True)[3]
        out = self.bn(out)
        return self.conv_seg(out)


def main(
    ds_path,
    model_name="dinov2_vitb14_reg",
    batch_size=16,
    num_workers=10,
    iters=80000,
    val_every=2000,
    seed=0,
    device="cuda",
):
    if (not device.startswith("cuda")) or (not torch.cuda.is_available()):
        device = "cpu"
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # set up data
    ds = ADE20K(ds_path, split="train")
    ds_val = ADE20K(ds_path, split="val")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    # set up model
    model = DINOLinear(model_name).to(device)
    # set up optimizer
    optim = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999)
    )
    sch1 = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-06, total_iters=1500
    )
    sch2 = torch.optim.lr_scheduler.PolynomialLR(
        optim, power=1, total_iters=40000 - 1500
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [sch1, sch2], [1500])

    st = time.time()
    iter_n = 0
    for sample in dl:
        if iter_n and (iter_n % val_every == 0):
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_sample in ds_val:
                    img, ann = val_sample
                    img, ann = img.to(device, non_blocking=True), ann.to(
                        device, non_blocking=True
                    )
                    img = img.permute(0, 3, 1, 2)
                    output = model(img)
                    output = torch.nn.functional.interpolate(
                        output, size=ann.shape[1:], mode="bilinear"
                    )
                    loss = torch.nn.functional.cross_entropy(
                        output, ann, reduction="none", ignore_index=255
                    )
                    val_loss += loss.sum() / ann.numel()
                print(f"Iteration {iter_n}/{iters}, Validation loss: {val_loss.item()}")
            model.train()
        # prepare data
        img, ann = sample
        img, ann = img.to(device, non_blocking=True), ann.to(device, non_blocking=True)
        img = img.permute(0, 3, 1, 2)
        # forward
        output = model(img)
        output = torch.nn.functional.interpolate(
            output, size=ann.shape[1:], mode="bilinear"
        )
        loss = torch.nn.functional.cross_entropy(
            output, ann, reduction="none", ignore_index=255
        )
        loss = loss.sum() / ann.numel()
        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        # log
        print(
            f"Iteration {iter_n}/{iters}, loss: {loss.item()}, speed {iter_n / (time.time()-st)}it/s",
            end="\r",
        )

        if iter_n == iters - 1:
            break
        else:
            iter_n += 1

    for _ in range(10):
        img, ann = ds[0]
        Image.fromarray(img).save("aimg.png")
        Image.fromarray(ann).save("aann.png")
        breakpoint()


if __name__ == "__main__":
    Fire(main)
