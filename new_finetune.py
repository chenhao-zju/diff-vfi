# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import re
import glob
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from pose_dataset import PoseData

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


from model import diffmae, diffusion
from engine_pretrain import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')


    parser.add_argument('--n_channels', type=int, default=7,    # 203
                            help='number of features, default=3')
    parser.add_argument('--emb_dim', type=int, default=1024,
                        help='feature dimension for embedding')
    parser.add_argument('--dec_emb_dim', type=int, default=512,
                        help='feature dimension for decoder embedding')

    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--decoder_num_heads', default=16, type=int)
    parser.add_argument('--depth', default=24, type=int)
    parser.add_argument('--decoder_depth', default=8, type=int)


    parser.add_argument('--mask_ratio', default=20, type=int,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--grad_weight', type=float, default=None,
                        help='weight decay (default: None)')
    parser.add_argument('--transl_weight', type=float, default=2.0,
                        help='the special transl weight comparing to pose and expression')
    parser.add_argument('--level', type=int, default=1,
                        help='the level of diff edge loss')
    parser.add_argument('--extra_edge', action='store_true', default=False,
                        help='using the extra edge loss or not')
    

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--fixed_number', type=int, default=100,
                        help='the length of one video')
    parser.add_argument('--start_index', type=int, default=10,
                        help='the start index of the input video')
    parser.add_argument('--train_part', default='pose',
                        help='the type of training part in this program')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    
    
    # define the model
    diff = diffusion.Diffusion(schedule='cosine')
    model = diffmae.DiffMAE(args, diff)

    model.to(device)
    print("Model = %s" % str(model))


    pattern = re.compile(r'\d+')
    file_list = sorted(glob.glob('{}/*.pth'.format(args.output_dir)), key=lambda x:int(pattern.findall(x)[-1]))

    checkpoint = torch.load(file_list[-1], map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)


    if args.eval:

        body_mean_std_pth = "/home/bingxing2/ailab/group/ai4earth/haochen/dataset/pose_dataset/anchor_trainposes/body_mean_std.json"

        f = open(body_mean_std_pth, 'r')
        content = f.read()
        body_mean_std = json.loads(content)

        body_mean = body_mean_std["body_mean"]
        body_mean = {key:np.array(value) for key, value in body_mean.items()}
        body_std = body_mean_std["body_std"]
        body_std = {key:np.array(value) for key, value in body_std.items()}

        test_path = "/home/bingxing2/ailab/group/ai4earth/haochen/dataset/pose_dataset/anchor_testposes/inputs.txt"
        with open(test_path, 'r') as f:
            pose_data = f.readlines()

        test_stats = evaluate(model, pose_data, epoch='', args=args, mean_std=[body_mean, body_std])

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
