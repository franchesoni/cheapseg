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
    def new_run_train_loop_fn(self): pass
    runner.run_train_loop = new_run_train_loop_fn.__get__(runner, type(runner))
    runner.train()
    # now we run our own `run_train_loop` 
    runner.call_hook('before_train')
    runner.call_hook('before_train_epoch')
    if runner.train_loop._iter > 0:
        print_log(
            f'Advance dataloader {runner.train_loop._iter} steps to skip data '
            'that has already been trained',
            logger='current',
            level=logging.WARNING)
        for _ in range(runner.train_loop._iter):
            next(runner.train_loop.dataloader_iterator)

    print('starting training')
    error_rates = []
    plot = False

    while runner.train_loop._iter < runner.train_loop._max_iters and not runner.train_loop.stop_training:
        print('iter', runner.train_loop._iter, ' '*20, end='\r')
        runner.model.train()

        batch = next(runner.train_loop.dataloader_iterator)
        # runner.train_loop.run_iter(data_batch)
        runner.call_hook('before_train_iter', batch_idx=runner.train_loop._iter, data_batch=batch)
        # outputs = self.runner.model.train_step(
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
            print('error rate', np.mean(error_rates))
            # sample a random click from the misclassified region
            err_region = misclassified.nonzero()
            assert len(err_region), 'no misclassified region found'
            click = err_region[torch.randint(0, err_region.shape[0], (1,))[0]]
            # get data at the click (feature, ground truth)
            patch_loc = click[0] // 14, click[1] // 14  # downsample according to dino patch size
            sample_x = feats[0, :, patch_loc[0], patch_loc[1]]  # [768]
            sample_y = int(seg_label[click[0], click[1]])  # [1]
            runner.model.decode_head.append(sample_x, sample_y)

            # LOGGING
            if plot:
                plt.imsave('sample_pred.png', pred_labels.cpu().numpy())
                plt.imsave('sample_gt.png', seg_label.cpu().numpy())
                plt.imsave('sample_misclassified.png', misclassified.cpu().numpy())

            if runner.train_loop._iter % 100 == 0:
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

        # runner.train_loop._decide_current_val_interval()
        # if ((runner.val_loop is not None)  and (runner.train_loop._iter >= runner.train_loop.val_begin)
        #         and (runner.train_loop._iter % runner.train_loop.val_interval == 0
        #                 or runner.train_loop._iter == runner.train_loop._max_iters or (runner.train_loop._iter in [2**i for i in range(5, 20)]))):
        #     runner.val_loop.run()

    runner.call_hook('after_train_epoch')
    runner.call_hook('after_train')
    runner.call_hook('after_run')

if __name__ == "__main__":
    main()


