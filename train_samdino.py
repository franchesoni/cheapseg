print("importing standard...")
import pickle
import traceback
import time
from pathlib import Path
import subprocess

print("importing external...")
import tqdm
import torch
import numpy as np
from fire import Fire
from torch.utils.tensorboard import SummaryWriter
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

print("importing local...")
from train_dino_linear import ADE20K, intersect_and_union, should_early_stop


print("done importing.")


class SamdinoDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path):
        super().__init__()
        ds_path = Path(ds_path)
        self.X_paths = sorted(
            ds_path.glob("*_X.pkl"), key=lambda fname: int(fname.name.split("_")[0])
        )
        self.y_paths = sorted(
            ds_path.glob("*_y.pkl"), key=lambda fname: int(fname.name.split("_")[0])
        )
        assert len(self.X_paths) == len(self.y_paths)

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        xpath = self.X_paths[idx]
        ypath = self.y_paths[idx]

        with open(xpath, "rb") as f:
            X = pickle.load(f)

        with open(ypath, "rb") as f:
            y = pickle.load(f)

        return X, y


def build_samdino_dataset(
    ds_path,
    out,
    iters=80000,
    batch_size=16,
    num_workers=10,
    dino_model_name="dinov2_vitb14_reg",
    sam_model_name="vit_b",
    device="cuda",
    seed=0,
    reset=False,
):
    if reset:
        import shutil

        shutil.rmtree(out)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # set device
    if (not device.startswith("cuda")) or (not torch.cuda.is_available()):
        device = "cpu"
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # set up data
    ds = ADE20K(ds_path, split="train")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    # set up sam
    sam = sam_model_registry[sam_model_name](checkpoint="sam_vit_b_01ec64.pth").to(
        device
    )
    mask_generator = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.7, stability_score_thresh=0.8, min_mask_region_area=7 * 7
    )
    # set up dino
    dino = torch.hub.load("facebookresearch/dinov2", dino_model_name).to(device)
    for param in dino.parameters():
        param.requires_grad = False

    # params for mean std normalization
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    ## go through the datasets
    finished, iter_n = False, 0
    st = time.time()
    while not finished:
        for sample in dl:
            time_so_far = time.time() - st
            print(
                f"Running {iter_n+1}/{iters}... ({max(iter_n / time_so_far, time_so_far/iter_n if iter_n else 0)} {'iter/s' if time_so_far < iter_n else 's/iter'})",
                end="\r",
            )
            img_batch, ann_batch = sample
            ann_batch = ann_batch.to(device)
            ds_maskss, targetss = [], []
            for batch_ind in range(len(img_batch)):
                img, ann = img_batch[batch_ind], ann_batch[batch_ind]
                masks = [
                    m["segmentation"]
                    for m in mask_generator.generate(img.numpy().astype(np.uint8))
                ]
                targets = []
                ignore_ind = []
                for m_ind, mask in enumerate(masks):
                    labels_in_mask = ann[mask]
                    area_per_label = torch.histc(
                        labels_in_mask.float(), bins=150, min=0, max=149
                    )
                    area_per_label = area_per_label / area_per_label.sum()
                    targets.append(area_per_label)
                targets = torch.stack(targets)
                targetss.append(targets.cpu())
                masks = torch.from_numpy(np.stack(masks)).to(device)
                ds_masks = torch.nn.functional.interpolate(
                    masks[:, None].float(), size=(37, 37), mode="area"
                )
                ds_maskss.append(ds_masks)
            img_batch = img_batch.to(device).permute(0, 3, 1, 2)

            with torch.no_grad():
                dino_feats = dino.get_intermediate_layers(
                    (img_batch - mean) / std, n=4, reshape=True
                )[3]
            featss = []
            for batch_ind in range(len(targetss)):
                ds_masks = ds_maskss[batch_ind]  # M,1,37,37
                feat = dino_feats[batch_ind]  # F, 37, 37
                avg_feat = torch.sum(
                    feat * ds_masks, dim=(2, 3), keepdims=True
                ) / torch.sum(
                    ds_masks, dim=(1, 2, 3), keepdims=True
                )  # M, F, 37, 37 -> M, F, 1, 1
                featss.append(avg_feat.squeeze().cpu())

            with open(out / f"{iter_n}_X.pkl", "wb") as f:
                pickle.dump(featss, f)
            with open(out / f"{iter_n}_y.pkl", "wb") as f:
                pickle.dump(targetss, f)
            iter_n += 1
            if iter_n == iters:
                finished = True
                break


class BNLinear(torch.nn.Module):
    def __init__(self, in_channels=768, n_classes=150):
        super().__init__()
        self.in_channels = in_channels
        self.bn = torch.nn.SyncBatchNorm(self.in_channels)
        self.linear = torch.nn.Linear(self.in_channels, n_classes)

    def forward(self, x):
        return self.linear(self.bn(x))


def main(
    ds_path,
    val_ds_path,
    batch_size=16,
    num_workers=10,
    iters=80000,
    val_every=200,
    seed=0,
    device="cuda",
    patience=None,
    sam_model_name="vit_b",
    dino_model_name="dinov2_vitb14_reg",
    val_only=2000,
):

    # set device
    if (not device.startswith("cuda")) or (not torch.cuda.is_available()):
        device = "cpu"
    # set logger
    writer = SummaryWriter()  # for visualization
    logfile = Path(f"{writer.log_dir}/samdino_{time.time()}.log")
    logfile.parent.mkdir(exist_ok=True)
    logfile.touch()  # create log file for record
    log_so_far = ""

    def update_log_file(iter_n, flush_every=50, force=False):
        if force or (iter_n and (iter_n % flush_every == 0)):
            with open(logfile, "w") as f:
                f.write(log_so_far)

    # log metadata
    current_git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    current_args = locals()
    log_so_far += f"Git hash: {current_git_hash}\n"
    log_so_far += f"Command line arguments: {current_args}\n"
    update_log_file(0, force=True)
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # set up data
    ds = SamdinoDataset(ds_path)
    ds_val = ADE20K(val_ds_path, split="val")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )
    # set up model
    model = BNLinear(in_channels=768).to(device)
    # set up sam
    sam = sam_model_registry[sam_model_name](checkpoint="sam_vit_b_01ec64.pth").to(
        device
    )
    mask_generator = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.7, stability_score_thresh=0.8, min_mask_region_area=7 * 7
    )
    # set up dino
    dino = torch.hub.load("facebookresearch/dinov2", dino_model_name).to(device)
    for param in dino.parameters():
        param.requires_grad = False
    # params for mean std normalization
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

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
    past_ious, finish = [], False
    while not finish:
        for sample in dl:
            if iter_n and (iter_n % val_every == 0):
                val_st = time.time()
                model.eval()
                with torch.no_grad():
                    print("Validating...")
                    intersections, unions = [], []
                    for val_sample_ind in tqdm.tqdm(range(len(ds_val))):
                        if val_sample_ind == val_only:
                            break
                        img, ann = ds_val[val_sample_ind]
                        H, W = img.shape[:2]
                        ann = torch.from_numpy(ann).to(device)
                        pred = torch.zeros((150, H, W)).to(device)
                        masks = [
                            m["segmentation"]
                            for m in mask_generator.generate(img.astype(np.uint8))
                        ]
                        masks = torch.from_numpy(np.stack(masks)).to(device)
                        ds_masks = torch.nn.functional.interpolate(
                            masks[:, None].float(), size=(H // 14, W // 14), mode="area"
                        )
                        img_batch = (
                            torch.from_numpy(img).to(device)[None].permute(0, 3, 1, 2)
                        )
                        with torch.no_grad():
                            dino_feats = dino.get_intermediate_layers(
                                (img_batch - mean) / std, n=4, reshape=True
                            )[3]
                        feat = dino_feats[0]
                        avg_feat = (
                            torch.sum(feat * ds_masks, dim=(2, 3), keepdims=True)
                            / torch.sum(ds_masks, dim=(1, 2, 3), keepdims=True)
                        ).squeeze()
                        pred_areas = model(avg_feat)  # M, C
                        for m_ind in range(len(masks)):
                            mask = masks[m_ind]
                            pred[:, mask] += pred_areas[m_ind].view(
                                150, 1
                            )  # add the uniform sampling bias
                        intersection, union, _, _ = intersect_and_union(
                            torch.argmax(pred, dim=0), ann, 150, 255
                        )
                        intersections.append(intersection)
                        unions.append(union)
                    IoU_per_class = torch.stack(intersections).sum(0) / torch.stack(
                        unions
                    ).sum(0)
                    mIoU = torch.nanmean(IoU_per_class)
                    past_ious.append(mIoU.item())
                    early_stop_now = should_early_stop(past_ious, patience)
                    # log
                    val_log_line = f"Iteration {iter_n}/{iters}, mIoU: {mIoU.item()}, val time: {time.time()-val_st}"
                    print(val_log_line)
                    log_so_far += val_log_line + "\n"
                    update_log_file(iter_n, force=True)
                    writer.add_scalar("val_mIoU", mIoU.item(), iter_n)
                    if early_stop_now:
                        print("Early stopping.")
                        log_so_far += "Early stopped."
                        update_log_file(iter_n, force=True)
                        finish = True
                        break
                model.train()
            update_log_file(iter_n)
            try:
                # prepare data
                X = torch.concatenate(
                    [x for batch_element in sample for x in batch_element[0]], dim=0
                )
                y = torch.concatenate(
                    [x for batch_element in sample for x in batch_element[1]], dim=0
                )
                nanmask = torch.isnan(X).any(dim=1) | torch.isnan(y).any(dim=1)
                X = X[~nanmask]
                y = y[~nanmask]
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                # forward
                output = model(X)
                loss = torch.nn.functional.l1_loss(output, y)
                # backprop
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
                # log
                writer.add_scalar("train_loss", loss.item(), iter_n)
                current_lr = optim.param_groups[0]["lr"]
                writer.add_scalar("lr", current_lr, iter_n)
                train_log_line = f"Iteration {iter_n}/{iters}, loss: {loss.item()}, lr: {current_lr}, speed: {iter_n / (time.time()-st)}it/s"
                print(
                    train_log_line,
                    end="\r",
                )
                log_so_far += train_log_line + "\n"
            except RuntimeError as e:
                print(f"runtime error at iter {iter_n}: {e}")
                traceback.print_exc()
                breakpoint()

            if iter_n == iters - 1:
                finish = True
                break
            else:
                iter_n += 1

    # completed!
    (Path(writer.log_dir) / "completed").touch()  # write completed file


if __name__ == "__main__":
    Fire({"build": build_samdino_dataset, "train": main})
