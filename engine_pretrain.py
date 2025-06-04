# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import math
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from PIL import Image

from einops import rearrange

import torch
from torchvision.transforms.functional import to_pil_image

import util.misc as misc
import util.lr_sched as lr_sched
from util.loss import calc_for_diffmae

import pose_dataset

def concat_images_horizontally(image_list):
    widths, heights = zip(*(img.size for img in image_list))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in image_list:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_img

def denormalize(tensor, mean, std):
    tensor = tensor * std + mean
    return tensor

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, iter=0):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

         # samples = samples.to(args.device, non_blocking=True)
        if args.train_part == 'body':
            samples = samples.to(args.device, non_blocking=True, dtype = torch.float32)
            samples = rearrange(samples, 'b n l c -> (b n) l c')
        else:
            poses, expressions = map(lambda x: x.to(args.device, non_blocking=True, dtype = torch.float32), samples) 
            poses = rearrange(poses, 'b n l c -> (b n) l c')
            expressions = rearrange(expressions, 'b n l c -> (b n) l c')


        with torch.cuda.amp.autocast():
            if args.train_part == 'body':
                pred, loss, ids_restore, mask, ids_masked, ids_keep = model( samples, weight=args.transl_weight )
            elif args.train_part == 'pose':
                pred, loss, ids_restore, mask, ids_masked, ids_keep = model( poses, weight=args.transl_weight )
            else:
                pred, loss, ids_restore, mask, ids_masked, ids_keep = model( expressions, weight=args.transl_weight )


        if args.grad_weight == None:
            loss_value = loss.item()
        else:
            loss, recons_loss, grad_loss = loss
            # loss = loss.mean()
            # recons_loss = recons_loss.mean()
            # grad_loss = grad_loss.mean()

            loss_value = loss.item()
            recons_loss = recons_loss.item()
            grad_loss = grad_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(loss=loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if args.grad_weight != None:
            metric_logger.update(recons_loss=recons_loss)
            metric_logger.update(grad_loss=grad_loss)
            recons_loss_value = misc.all_reduce_mean(recons_loss)
            grad_loss_value = misc.all_reduce_mean(grad_loss)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)

            if args.grad_weight != None:
                log_writer.add_scalar('recons_loss', recons_loss_value, epoch_1000x)
                log_writer.add_scalar('grad_loss', grad_loss_value, epoch_1000x)

            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model: torch.nn.Module,
                    data_loader: Iterable,
                    epoch='',
                    log_writer=None,
                    args=None,
                    mean_std=None):
    
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test: '
    print_freq = 20

    body_mean, body_std = mean_std

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader[:-2], print_freq, header)):

        pth, start_index, start_video1, start_video2 = samples.split(' ')
        start_video = int(start_video2)
        start_index = int(start_index) - int(start_video)


        name = pth.split('/')[-4]
        body,_ , all_dataset = pose_dataset.read_pkl(pth, True)
        print(body.shape)
        normalized_body = (body[start_video: start_video+args.fixed_number] - body_mean[name]) / body_std[name]
        normalized_body = torch.tensor(normalized_body).unsqueeze(0).to(device=args.device, dtype=torch.float32)


        with torch.cuda.amp.autocast():
            pred, loss, ids_restore, mask, ids_masked, ids_keep = model( normalized_body, weight=args.transl_weight )
            
        

        if args.grad_weight == None:
            loss_value = loss.item()
        else:
            loss, recons_loss, grad_loss = loss
            loss_value = loss.item()
            recons_loss = recons_loss.item()
            grad_loss = grad_loss.item()


        loss /= accum_iter

        metric_logger.update(loss=loss_value)

        visible_tokens = torch.gather(normalized_body, dim=1, index=ids_keep[:, :, None].expand(-1, -1, normalized_body.shape[2]))

        model_ = model

        for n in range(pred.size()[0]):
            if n % 100 == 0:
                sampled_token = model_.diffusion.sample(pred[n].unsqueeze(0))
                sampled_token = sampled_token.squeeze()

                img = torch.cat([visible_tokens[n], sampled_token], dim=0)
                img = torch.gather(img, dim=0, index=ids_restore[n].unsqueeze(-1).repeat(1, img.shape[1])) # to unshuffle

                img = img.detach().cpu().numpy()
                img = denormalize(img, body_mean[name], body_std[name])

                pth_list = pth.split('/')
                pth_list[-1] = 'body_NewLR_WeightedLoss_trans_extraloss3_translweight5_multilevel_len100mask40_onlyglobal_0602_100.pkl'
                pkl_pth = '/' + '/'.join(pth_list)

                pose_dataset.write_body_pkl_100(img.squeeze(), all_dataset, start_video, start_index, args.fixed_number, args.mask_ratio, pkl_pth)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
