print("importing standard...")
import pickle
import shutil
import traceback
import time
from pathlib import Path
import subprocess

print("importing external...")
import matplotlib.pyplot as plt
from PIL import Image
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

def build_simple_samdino_dataset(
    ds_path,
    out,
    num_workers=10,
    dino_model_name="dinov2_vitb14_reg",
    sam_model_name="vit_b",
    device="cuda",
    seed=0,
    reset=False,
):
    """We build a simple dataset out of ADE20K. For each image we extract SAM masks. For each SAM mask we save an image containing the mask, the mask, a DINOv2+reg embedding, and the percentage of each class inside the mask."""
    print('init...', end='\r')
    if reset:
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
    print('data...', end='\r')
    ds_path = Path(ds_path)
    img_dir = ds_path / "images" / "training"
    ann_dir = ds_path / "annotations" / "training"
    img_files = sorted(img_dir.glob("*.jpg"))
    ann_files = sorted(ann_dir.glob("*.png"))
    assert len(img_files) == len(
        ann_files
    ), "Mismatched number of images and annotations"
    assert (img_files[0].stem == ann_files[0].stem) and (
        img_files[-1].stem == ann_files[-1].stem
    ), "Mismatched files"
    samples = list(zip(img_files, ann_files))

    # set up sam
    print('sam...', end='\r')
    sam = sam_model_registry[sam_model_name](checkpoint="sam_vit_b_01ec64.pth").to(
        device
    )
    mask_generator = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.7, stability_score_thresh=0.8, min_mask_region_area=7 * 7
    )
    # set up dino
    print('dino...', end='\r')
    dino = torch.hub.load("facebookresearch/dinov2", dino_model_name).to(device)
    for param in dino.parameters():
        param.requires_grad = False

    # params for mean std normalization
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)


    # 1. generate masks, 2. extract bboxes, 3. generate image crops, 4. generate mask crops, 5. generate targets, 6. create DINOv2reg embeddings, 7. save sample.
    for idx in tqdm.tqdm(range(len(samples))):
        img_path, ann_path = samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        ann = torch.from_numpy(np.array(Image.open(ann_path)))
        # handle background as in mmseg
        ann[ann == 0] = 255
        ann = ann - 1
        ann[ann == 254] = 255



        print(f'generating {idx}...', end='\r')
        # 1. generate masks
        masks = mask_generator.generate(img)
        # 2. extract bboxes
        bboxes = [m['bbox'] for m in masks]
        bboxes = [(max(bbox[0]-bbox[2]//8,0), max(bbox[1]-bbox[3]//8, 0), min(bbox[2]+bbox[2]//4, img.shape[1]), min(bbox[3]+bbox[3]//4, img.shape[0])) for bbox in bboxes]
        # 3. image crops
        img_crops = [img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] for bbox in bboxes]
        # 4. masks and crops
        segs = [m["segmentation"] for m in masks]
        mask_crops = [seg[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] for seg, bbox in zip(segs, bboxes)]

        # 5. generate targets
        targets = []
        for mask in segs:
            labels_in_mask = ann[mask]
            area_per_label = torch.histc(labels_in_mask.float(), bins=150, min=0, max=149)
            area_per_label = area_per_label / area_per_label.sum()
            targets.append(area_per_label)

        # 6. DINOv2reg embeddings
        emb = []
        with torch.no_grad():
            for img_crop in img_crops:
                pad_h = 14 - img_crop.shape[0] % 14
                pad_w = 14 - img_crop.shape[1] % 14
                crop = np.pad(
                    img_crop,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                dino_feats = dino.forward_features(
                    (torch.from_numpy(crop).permute(2, 0, 1)[None].float().to(device) - mean) / std, 
                )
                emb.append(dino_feats)

        # 7. save sample
        sample = {
            "img_crops": img_crops,
            "mask_crops": mask_crops,
            "targets": targets,
            "emb": emb,
        }

        breakpoint()
        plt.imsave('tmp0.png', img)
        for idx in range(len(img_crops)):
            plt.imsave('tmp1.png', img_crops[idx])
            plt.imsave('tmp2.png', mask_crops[idx])
            print(np.argsort(np.array(targets[idx]))[::-1][:5])

        # np.save(out / f"sample_{str(idx).zfill(5)}.npy", sample)



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





def val_oracle(val_ds_path, sam_model_name='vit_b', val_only_n=2000, vis=False, device='cuda', effvit=False):
    # set device
    if (not device.startswith("cuda")) or (not torch.cuda.is_available()):
        device = "cpu"
    ds_val = ADE20K(val_ds_path, split="val")
    # set up sam
    if not effvit:
        sam = sam_model_registry[sam_model_name](checkpoint="sam_vit_b_01ec64.pth").to(
            device
        ).eval()
        mask_generator = SamAutomaticMaskGenerator(
            sam, pred_iou_thresh=0.7, stability_score_thresh=0.8, min_mask_region_area=7 * 7
        )
    else:
        # segment anything
        from efficientvit.sam_model_zoo import create_sam_model
        from efficientvit.efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator
        efficientvit_sam = create_sam_model(
        name="l2", weight_url="efficientvit/assets/checkpoints/sam/l2.pt",
        )
        efficientvit_sam = efficientvit_sam.to(device).eval()
        mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, pred_iou_thresh=0.7, stability_score_thresh=0.8, min_mask_region_area=7*7)
    print("Validating...")
    intersections, unions = [], []
    intersections2, unions2 = [], []
    # for val_sample_ind in tqdm.tqdm(range(len(ds_val))):
    val_st = time.time()
    for val_sample_ind in range(len(ds_val)):
        if val_sample_ind == val_only_n:
            break
        print(f'{val_sample_ind+1}/{len(ds_val)} loading data'+' '*20, end='\r')

        img, ann = ds_val[val_sample_ind]
        H, W = img.shape[:2]
        ann = torch.from_numpy(ann).to(device)
        print(f'{val_sample_ind+1}/{len(ds_val)} sam generator'+' '*20, end='\r')
        masks = [
            m["segmentation"]
            for m in mask_generator.generate(img.astype(np.uint8))
        ]
        masks = np.stack(masks)


        if vis:
            print(f'{val_sample_ind+1}/{len(ds_val)} visualizing'+' '*20, end='\r')
            if Path('vis').exists():
                shutil.rmtree('vis')
                Path('vis').mkdir()

            Image.fromarray(img.astype(np.uint8)).save('vis/img.png')
            masks_intersection = np.sum(masks[:, None] * masks[None], axis=(2,3))
            masks_union = np.sum(masks[:, None]*1 + masks[None], axis=(2,3)) - masks_intersection
            masks_overlap = masks_intersection / masks_union
            current_idx, remaining_idx, vis_idx = 0, set(list(range(len(masks)))), 0
            while len(remaining_idx):
                Image.fromarray(masks[current_idx]).save(f"vis/mask_{str(vis_idx).zfill(len(str(len(masks))))}_{current_idx}.png")
                vis_idx += 1
                remaining_idx.remove(current_idx)
                masks_overlap[:, current_idx] = 0
                for new_idx in np.argsort(masks_overlap[current_idx])[::-1]:
                    if new_idx in remaining_idx:
                        break
                current_idx = new_idx
        masks = torch.from_numpy(masks).to(device)

        print(f'{val_sample_ind+1}/{len(ds_val)} target computation'+' '*20, end='\r')
        targets = []
        for m_ind, mask in enumerate(masks):
            labels_in_mask = ann[mask]
            area_per_label = torch.histc(
                labels_in_mask.float(), bins=150, min=0, max=149
            )
            denom = area_per_label.sum()
            area_per_label = area_per_label / denom if denom else area_per_label
            targets.append(area_per_label)
        pred_areas = torch.stack(targets)

        print(f'{val_sample_ind+1}/{len(ds_val)} prediction computation'+' '*20, end='\r')
        pred = torch.zeros((150, H, W)).to(device)
        pred_maj = torch.zeros((150, H, W)).to(device)
        for m_ind in range(len(masks)):
            mask = masks[m_ind]
            pred[:, mask] += pred_areas[m_ind].view(
                150, 1
            )  # add the rel freq of each class in those pixels
            if (pred_areas[m_ind] > 0).any():
                pred_maj[torch.argmax(pred_areas[m_ind]), mask] += 1  # just sum masks, each mask classified as their majority class

        print(f'{val_sample_ind+1}/{len(ds_val)} score computation'+' '*20, end='\r')
        intersection, union, _, _ = intersect_and_union(
            torch.argmax(pred, dim=0), ann, 150, 255
        )
        intersection_maj, union_maj, _, _ = intersect_and_union(
            torch.argmax(pred_maj, dim=0), ann, 150, 255
        )
        intersections.append(intersection)
        unions.append(union)
        intersections2.append(intersection_maj)
        unions2.append(union_maj)
        IoU_per_class = torch.stack(intersections).sum(0) / torch.stack(
            unions
        ).sum(0)
        IoU_per_class2 = torch.stack(intersections2).sum(0) / torch.stack(
            unions2
        ).sum(0)
        mIoU = torch.nanmean(IoU_per_class)
        mIoU2 = torch.nanmean(IoU_per_class2)
        print('Area oracle mIoU:', mIoU, 'Maj oracle mIoU', mIoU2, 'eta (s):', (time.time()-val_st)/(val_sample_ind+1)*len(ds_val))
    print('Oracle final mIoU (area-based):', mIoU)
    print('Oracle final mIoU (majority-based):', mIoU2)

    

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
            if (iter_n % val_every == 0):
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
                        masks = np.stack(masks)
                        breakpoint()
                        from PIL import Image
                        Image.fromarray(img.astype(np.uint8)).save('vis/img.png')
                        masks_intersection = np.sum(masks[:, None] * masks[None], axis=(2,3))
                        masks_union = np.sum(masks[:, None]*1 + masks[None], axis=(2,3)) - masks_intersection
                        masks_overlap = masks_intersection / masks_union
                        current_idx, remaining_idx, vis_idx = 0, set(list(range(len(masks)))), 0
                        while len(remaining_idx):
                            Image.fromarray(masks[current_idx]).save(f"vis/mask_{str(vis_idx).zfill(len(str(len(masks))))}_{current_idx}.png")
                            vis_idx += 1
                            remaining_idx.remove(current_idx)
                            masks_overlap[:, current_idx] = 0
                            for new_idx in np.argsort(masks_overlap[current_idx])[::-1]:
                                if new_idx in remaining_idx:
                                    break
                            current_idx = new_idx
                        masks = torch.from_numpy(masks).to(device)


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

def visualize_simple():
    import matplotlib.pyplot as plt
    dspath = Path("simpledinosamds")
    for sample_path in sorted(dspath.glob('*.npy')):
        sample = np.load(sample_path, allow_pickle=True).item()
        crops = sample["img_crops"]
        mask_crops = sample["mask_crops"]
        targets = sample["targets"]
        for idx in range(len(crops)):
            plt.imsave('tmp1.png', crops[idx])
            plt.imsave('tmp2.png', mask_crops[idx])
            print(np.argsort(np.array(targets[idx]))[::-1][:5])
            breakpoint()

class SimpleSamdinoDataset(torch.utils.data.Dataset):
    def __init__(self, dspath):
        dspath = Path(dspath)
        self.samples = sorted(dspath.glob('*.npy'))

    def __getitem__(self, idx):
        sample = np.load(self.samples[idx], allow_pickle=True).item()
        return sample

    def __len__(self):
        return len(self.samples)

def train_simple_sam_classifier(n_epochs=12):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set up dino
    dino = torch.hub.load("facebookresearch/dinov2", 'dinov2_vitb14_reg').to(device)
    for param in dino.parameters():
        param.requires_grad = False

    # params for mean std normalization
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    writer = SummaryWriter()
    ds = SimpleSamdinoDataset('simpleds')
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    global_step = 0
    for epoch in range(n_epochs):
        for sample in dl:
            breakpoint()

    


if __name__ == "__main__":
    Fire({"simple": build_simple_samdino_dataset, "build": build_samdino_dataset, "train": main, "oracle": val_oracle, "vis":visualize_simple, "samcls": train_simple_sam_classifier})
    # use `python train_samdino.py simple` to generate the dataset
    # use `python train_samdino.py samcls` to train a sam classifier
