import os.path as osp
import logging

import torch
import torch.utils

from train import parse_args, Runner, Config
from mmengine.logging import print_log

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

def pct_norm(x, pct=1):
    newmin = np.percentile(x, pct)
    newmax = np.percentile(x, 100 - pct)
    return np.clip((x - newmin) / (newmax - newmin), 0, 1)



def preprocess_image_array_dino(image_array, target_size, device):
    # Step 1: Normalize using mean and std of ImageNet dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array / 255.0 - mean) / std

    # Step 2: Resize the image_array to the target size
    image_array = np.transpose(
        image_array, (2, 0, 1)
    )  # PyTorch expects (C, H, W) format
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = resize_image_tensor(image_tensor, target_size=target_size).to(device)
    return image_tensor

def resize_image_tensor(image_tensor, target_size):
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
        ]
    )
    resized_image = transform(image_tensor)
    return resized_image






def describe(x):
    outstr = ''
    things = [len, type, lambda x: x.shape, lambda x: x.keys()]
    names = ['len', 'type', 'shape', 'keys']
    for thing, name in zip(things, names):
        try:
            answer = name + ' ' + str(thing(x)) + ', '
        except:
            answer = ''
        outstr += answer
    print(outstr)

class CLSDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.feats = []
        self.gt = []

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.gt[idx]

    def update(self, feat, gt):
        self.feats.append(feat)
        self.gt.append(gt)
        return self

def compute_augclick(click, augcfg, orig_scale_factor):
    scale_factor_data = np.array(orig_scale_factor).reshape(2)  # the one used to generate the current image (no aug)
    augclick = click.cpu() / scale_factor_data   # click on original image
    scale_factor_aug = np.array(augcfg['scale_factor']).reshape(2)
    augclick = augclick * scale_factor_aug  # click on augmented image
    augclick = torch.minimum(augclick, torch.tensor(augcfg['img_shape_before_crop']) - 1)  # clip to image size
    crop_bbox = augcfg['crop_bbox']
    if augclick[0] < crop_bbox[0] or augclick[1] < crop_bbox[2] or augclick[0] >= crop_bbox[1] or augclick[1] >= crop_bbox[3]:
        return None
    augclick[0], augclick[1] = augclick[0] - crop_bbox[0], augclick[1] - crop_bbox[2]  # click on cropped image
    shape_after_crop = augcfg['img_shape_after_crop']
    assert augclick[0] < shape_after_crop[0] and augclick[1] < shape_after_crop[1], f'click {augclick} not in {shape_after_crop}, augcfg {augcfg}, click {click}, scale_factor_data {scale_factor_data}'
    augclick[1] = shape_after_crop[1] - augclick[1]  if augcfg['flip'] else augclick[1]  # flip y
    assert augclick[0] >= 0 and augclick[1] >= 0, f'click {augclick} negative, augcfg {augcfg}'
    return augclick


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.resume = args.resume
    assert not ('runner_type' in cfg)
    cfg['randomness'] = dict(seed=0)

    # initialize dataset with feats and gt only 
    clsds = CLSDataset()
    runner = Runner.from_cfg(cfg)
    # we call `train` to init everything, but we remove the last part of the train fn to use our custom loop instead
    def dummy_run_loop_fn(self): pass
    runner.run_train_loop = dummy_run_loop_fn.__get__(runner, type(runner))
    runner.train()
    # now we run our own `run_train_loop` 
    runner.call_hook('before_train')
    runner.call_hook('before_train_epoch')
    print('starting training')
    error_rates = []
    plot = False
    tag = 'debug'
    n_aug = 8

    for idx, batch in enumerate(runner.test_dataloader):  # in fact this loads train data without augmentation
        if idx > 24500:
            break
        runner.model.train()

        # substitute: runner.train_loop.run_iter(data_batch)
        runner.call_hook('before_train_iter', batch_idx=runner.train_loop._iter, data_batch=batch)
        breakpoint()
        # substitute: outputs = self.runner.model.train_step(
        #     data_batch, optim_wrapper=self.runner.optim_wrapper)
        with torch.no_grad():
            # IGNORE
            # # used to see if dinov2 output was the same (it kinda is)
            # intensordino = preprocess_image_array_dino(inputimg, (518, 518), device=batch['inputs'][0].device)
            # batch['inputimg'] = inputimg
            # batch['intensordino'] = intensordino

            # FEAT EXTRACTION
            data_batch = runner.model.data_preprocessor(batch, True)
            intensor = data_batch['inputs']  # 1, 3, 518, 518
            feats = list(runner.model.extract_feat(intensor))  # list with 4 feat maps [1, 768, 37, 37]

            # IGNORE
            # # used to see if dinov2 output was the same (it kinda is)
            # dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(intensor.device)
            # dinofeats = dino.forward_features(intensordino.to(intensor.device))["x_norm_patchtokens"]
            # # dinofeats is (1, 37**2, 768) and feats[3] is (1, 768, 37, 37)

            # LOGGING
            if plot:
                mmfeats = feats[3][0].permute(1,2,0).reshape(1, 37**2, 768)
                allfeats = mmfeats
                # allfeats = torch.concatenate((mmfeats, dinofeats), dim=0)  # (2, 37**2, 768)
                pca_all = PCA(n_components=3).fit_transform(allfeats.reshape(-1, 768).cpu().numpy()).reshape(-1, allfeats.shape[1], 3).reshape(-1, 37, 37, 3)  # (2, 37**2, 3)
                pca_all = np.concatenate(list(pca_all), axis=1)
                plt.imsave('sample_pca.png', pct_norm(pca_all))
                inputimg = batch['inputs'][0].permute(1,2,0).numpy()[..., ::-1]  # debug
                plt.imsave('sample_img.png', inputimg)
                plt.imsave('sample_input.png', pct_norm(data_batch['inputs'][0].permute(1,2,0).cpu().numpy()))

            # PREDICT
            feats = feats[3] / torch.norm(feats[3], dim=1, keepdim=True)  # normalize [1, 768, 37, 37]
            prediction = runner.model.decode_head.forward(feats)

###### ALL OF THIS WAS FOR PROTOTYPES:
        #     # decode with linear layer
        #     seg_logits = runner.model.decode_head.forward(feats)  # [1, 150*P, 37, 37]
        #     # aggregate prototypes (from 150*P to 150)
        #     log_probs = torch.logsumexp(seg_logits.view(1, 150, cfg.n_prototypes, 37, 37), dim=2) - torch.logsumexp(seg_logits, dim=1, keepdim=True)
        #     # upsample
        #     log_probs = torch.nn.functional.interpolate(log_probs, size=(518, 518), mode='bilinear', align_corners=False)
            log_probs = torch.nn.functional.interpolate(prediction, size=(518, 518), mode='bilinear', align_corners=False)
            # make predictions and compute err region
            pred_labels = torch.argmax(log_probs, dim=1)[0]  # [518, 518]

            # COMPARE and CLICK
            seg_label = runner.model.decode_head._stack_batch_gt(data_batch['data_samples'])[0][0]  # [518, 518]
            uniques = torch.unique(seg_label)
            if (len(uniques) == 1) and (uniques == 255):
                print('no gt found iter', runner.train_loop._iter)
                continue
            misclassified = pred_labels != seg_label
            misclassified[seg_label==255] = False  # ignore
            error_rate = misclassified.float().sum() / (518*518 - (seg_label==255).sum())
            error_rates.append(float(error_rate))
            print('iter', runner.train_loop._iter, 'error rate', np.mean(error_rates), end='\r')
            # sample a random click from the misclassified region
            err_region = misclassified.nonzero()
            if len(err_region) == 0:
                print('no cls error region found')
                continue
            click = err_region[torch.randint(0, err_region.shape[0], (1,))[0]]

            if n_aug > 0:
                # generate naug augmented versions of the input that contain the click
                sample_idx = batch['original_pipeline']['sample_idx'][0]
                n = 0
                augsamples, augclicks = [], []
                while n < n_aug:
                    # generate one image
                    augsample = runner.train_dataloader.dataset[sample_idx]
                    # for the dataset return value to be equivalent to usual train batch we need nest a list 
                    augsample = augsample | {'inputs': [augsample['inputs']], 'data_samples': [augsample['data_samples']]}
                    augsample = runner.model.data_preprocessor(augsample, True)

                    # geometric pipeline is: resize, pad
                    # for aug is: random resize, random crop, random flip, pad
                    augclick = compute_augclick(click, augsample['original_pipeline'], data_batch['original_pipeline']['scale_factor'])
                    if augclick is None:
                        continue

                    augsamples.append(augsample['inputs'][0])
                    augclicks.append(augclick)
                    n += 1
                    
                augsamples = torch.stack(augsamples)
                augfeats = list(runner.model.extract_feat(augsamples))
                augfeats = augfeats[3] / torch.norm(augfeats[3], dim=1, keepdim=True)  # normalize [B, 768, 37, 37]
            else:
                augclicks = []
                augfeats = torch.Tensor().to(feats.device)

            # get data at the click (feature, ground truth)
            sample_y = int(seg_label[click[0], click[1]])  # [1]
            for clk, ff in zip([click] + augclicks, list(torch.cat((feats, augfeats), dim=0))):
                patch_loc = int(clk[0] // 14), int(clk[1] // 14)  # downsample according to dino patch size
                sample_x = ff[:, patch_loc[0], patch_loc[1]]  # [768]
                runner.model.decode_head.append(sample_x, sample_y)
            if 'sqrt' in tag:
                runner.model.decode_head.k = int((len(runner.model.decode_head.labels)/(n_aug+1))**0.5)

            # LOGGING
            if plot:
                plt.imsave('sample_pred.png', pred_labels.cpu().numpy())
                plt.imsave('sample_gt.png', seg_label.cpu().numpy())
                plt.imsave('sample_misclassified.png', misclassified.cpu().numpy())

            if runner.train_loop._iter % 500 == 0:
                np.save(f'sample_error_rate{tag}.npy', np.array(error_rates))
                plt.figure()
                plt.plot(error_rates)
                plt.xlabel('iteration')
                plt.ylabel('error rate')
                plt.savefig('sample_error_rate.png')
                # now the same but smoothed
                plt.figure()
                plt.plot(np.convolve(error_rates, np.ones(100)/100, mode='valid'))
                plt.xlabel('iteration')
                plt.ylabel('error rate')
                plt.savefig('sample_error_rate_smoothed.png')

        # # finetune decoder
        # clsdl = torch.utils.data.DataLoader(clsds, batch_size=32, num_workers=cfg.train_dataloader.num_workers, shuffle=True)
        # iteration = 0
        # MAX_ITERS = 1024
        # while iteration < MAX_ITERS:
        #     for xybatch in clsdl:
        #         if iteration >= MAX_ITERS:
        #             break
        #         print('subtrain iteration', iteration, end='\r')
        #         feats, gts = xybatch
        #         with runner.optim_wrapper.optim_context(runner.model):
        #             logits = runner.model.decode_head.forward(feats)
        #             log_probs = torch.logsumexp(logits.view(logits.shape[0], 150, cfg.n_prototypes, logits.shape[2], logits.shape[3]), dim=2) - torch.logsumexp(logits, dim=1, keepdim=True)
        #             losses = {'loss_ce': torch.nn.functional.cross_entropy(log_probs.view(log_probs.shape[0], 150), gts)}
        #         parsed_losses, log_vars = runner.model.parse_losses(losses)
        #         runner.optim_wrapper.update_params(parsed_losses)
        #         iteration += 1

        log_vars = {}
        runner.call_hook('after_train_iter', batch_idx=runner.train_loop._iter, data_batch=batch, outputs=log_vars)
        runner.train_loop._iter += 1

        runner.train_loop._decide_current_val_interval()
        if ((runner.val_loop is not None)  and (runner.train_loop._iter >= runner.train_loop.val_begin)
                and (runner.train_loop._iter % runner.train_loop.val_interval == 0
                        or runner.train_loop._iter == runner.train_loop._max_iters or (runner.train_loop._iter in [8**i for i in range(2, 20)]))):
            runner.val_loop.run()
            print('finished validating at step', runner.train_loop._iter)

    runner.call_hook('after_train_epoch')
    runner.call_hook('after_train')
    runner.call_hook('after_run')

if __name__ == "__main__":
    main()



# 09/21 00:17:58 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 41.0800  mIoU: 2.2400  mAcc: 4.2600  data_time: 0.0018  time: 0.0268
# finished validating at step 64

# 09/21 00:19:02 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 57.2100  mIoU: 7.2900  mAcc: 11.2600  data_time: 0.0017  time: 0.0271
# finished validating at step 512

# 09/21 00:21:19 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 66.8600  mIoU: 15.6100  mAcc: 20.7100  data_time: 0.0026  time: 0.0292
# finished validating at step 4096

# 09/21 00:06:40 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 68.4300  mIoU: 19.4800  mAcc: 25.0800  data_time: 0.0022  time: 0.0289
# finished validating at step 8000

#  aAcc: 70.0600  mIoU: 23.9700  mAcc: 30.5200  data_time: 0.0019  time: 0.0300
# finished validating at step 16000

# 09/21 00:15:12 - mmengine - INFO - Iter(val) [2000/2000]    aAcc: 70.8000  mIoU: 26.7400  mAcc: 33.9800  data_time: 0.0021  time: 0.0315
# finished validating at step 24000



