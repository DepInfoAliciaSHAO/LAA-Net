#-*- coding: utf-8 -*-
import os
import sys
import time
import argparse
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import math
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from package_utils.transform import (
    final_transform,
    get_center_scale, 
    get_affine_transform,
)
from configs.get_config import load_config
from models import *
from package_utils.utils import vis_heatmap
from package_utils.image_utils import load_image, crop_by_margin
from losses.losses import _sigmoid
from lib.metrics import get_acc_mesure_func, bin_calculate_auc_ap_ar
from datasets import DATASETS, build_dataset
from lib.core_function import AverageMeter
from logs.logger import Logger, LOG_DIR

from datetime import datetime
import csv

def parse_args(args=None):
    arg_parser = argparse.ArgumentParser('Processing testing...')
    arg_parser.add_argument('--cfg', '-c', help='Config file', required=True)
    arg_parser.add_argument('--image', '-i', type=str, help='Image for the single testing mode!')
    arg_parser.add_argument('--type', '-t', type =str, help = 'SBI or BI')
    args = arg_parser.parse_args(args)
    
    return args

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def asserts(task, args):
    if task == 'test_img':
        assert args.image is not None, "Image can not be None with single image test mode!"
        logger.info('Turning on single image test mode...')
    else:
        logger.info('Turning on evaluation mode...')
    if task == 'eval' and cfg.DATASET.DATA.TEST.FROM_FILE:
        assert cfg.DATASET.DATA.TEST.ANNO_FILE is not None, "Annotation file can not be None with evaluation test mode!"
        assert len(cfg.DATASET.DATA.TEST.ANNO_FILE), "Annotation file can not be empty with evaluation test mode!"

def get_model(cfg_model, logger_, pretrained_pth):
    model = build_model(cfg_model, MODELS).to(torch.float64)
    logger_.info('Loading weight ... {}'.format(pretrained_pth))
    return load_pretrained(model, pretrained_pth)

def get_args():
    if sys.argv[1:] is not None:
        args = sys.argv[1:]
    else:
        args = sys.argv[:-1]
    return parse_args(args)

def evaluation_test_image(image, aspect_ratio, pixel_std, rot, image_size, device_count, logger, test_vis_hm, test_threshold):
        img = load_image(image)
        img = cv2.resize(img, (317, 317))
        img = img[60:(317), 30:(287), :]
        c, s = get_center_scale(img.shape[:2], aspect_ratio, pixel_std)
        trans = get_affine_transform(c, s, rot, image_size)
        input = cv2.warpAffine(img,
                               trans,
                               (int(image_size[0]), int(image_size[1])),
                               flags=cv2.INTER_LINEAR,
                              )
        with torch.no_grad():
            st = time.time()
            img_trans = transforms(input/255).to(torch.float64)
            img_trans = torch.unsqueeze(img_trans, 0)
            if device_count > 0:
                img_trans = img_trans.cuda(non_blocking=True)
            
            outputs = model(img_trans)
            hm_outputs = outputs[0]['hm']
            cls_outputs = outputs[0]['cls']
            hm_preds = _sigmoid(hm_outputs).cpu().numpy()
            if test_vis_hm:
                print(f'Heatmap max value --- {hm_preds.max()}')
                vis_heatmap(img, hm_preds[0], 'output_pred.jpg')
            label_pred = cls_outputs.cpu().numpy()
            label = 'Fake' if label_pred[0][-1] > test_threshold else 'Real'
            logger.info('Inferencing time --- {}'.format(time.time() - st))
            logger.info('{} --- {}'.format(label, label_pred[0][-1]))
            logger.info('-----------------***--------------------')

def get_vid_probs(vid_preds):
    vid_probs = {}
    for k in vid_preds.keys():
        vid_probs[k] = vid_preds[k].sigmoid().cpu().numpy().flatten().tolist()
    return vid_probs

def save_as_csv(vid_probs, root, data_type, SBI):
    res_dir = os.path.join(root, 'test', data_type, "results")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_laa-net")
    if (SBI):
        now_str += "_sbi"
    else:
        now_str += "_bi"
    file_name = os.path.join(res_dir, now_str) + '.csv'
    # Find max number of values to create header dynamically
    max_len = max(len(v) for v in vid_probs.values())

    # Save to CSV
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = ["ID"] + [f"value_{i}" for i in range(max_len)]
        writer.writerow(header)

        for key, values in vid_probs.items():
            # Pad with empty strings if values are shorter
            padded_values = values + [""] * (max_len - len(values))
            writer.writerow([key] + padded_values)



if __name__=='__main__':

    args = get_args()
    
    # Loading config file
    cfg = load_config(args.cfg)
    logger = Logger(task='testing')

    #Seed
    set_seeds(cfg.SEED)

    # Define essential variables
    image = args.image; test_file = cfg.TEST.test_file; video_level = cfg.TEST.video_level
    aspect_ratio = cfg.DATASET.IMAGE_SIZE[1]*1.0 / cfg.DATASET.IMAGE_SIZE[0]
    pixel_std = 200; rot = 0;     metrics_base = cfg.METRICS_BASE
    
    transforms = final_transform(cfg.DATASET)
    acc_measure = get_acc_mesure_func(metrics_base)
    task = cfg.TEST.subtask; flip_test = cfg.TEST.flip_test
    logger.info('Flip Test is used --- {}'.format(flip_test))
    asserts(task, args)

    # build and load/initiate pretrained model
    model = get_model(cfg.MODEL, logger, cfg.TEST.pretrained)
    device_count = torch.cuda.device_count()
    if device_count >= 1:
        model = nn.DataParallel(model, device_ids=cfg.TEST.gpus).cuda()
    else:
        model = model.cuda()
    
    #Model evaluation
    model.eval()
    if image is not None and task == 'test_img':
        evaluation_test_image(image, aspect_ratio, rot, pixel_std, cfg.DATASET.IMAGE_SIZE, device_count, logger, cfg.TEST.vis_hm, cfg.TEST.threshold)
    if task == 'eval':
        logger.info(f'Using metric-base {metrics_base} for evaluation!')
        logger.info(f'Video level evaluation mode: {video_level}')
        st = time.time()
        test_dataset = build_dataset(cfg.DATASET, 
                                     DATASETS,
                                     default_args=dict(split='test', config=cfg.DATASET))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                     shuffle=False,
                                     num_workers=cfg.DATASET.NUM_WORKERS)
        logger.info('Dataset loading time --- {}'.format(time.time() - st))

        # Make sure all tensors in same device
        total_preds = torch.tensor([]).cuda().to(dtype=torch.float64)
        total_labels = torch.tensor([]).cuda().to(dtype=torch.float64)
        vid_preds = {}
        vid_labels = {}

        acc = AverageMeter(); auc = AverageMeter(); ar = AverageMeter(); ap = AverageMeter()
        test_dataloader = tqdm(test_dataloader, dynamic_ncols=True)

        with torch.no_grad():
            for b, (inputs, labels, vid_ids) in enumerate(test_dataloader):
                i_st = time.time()
                if device_count > 0:
                    inputs = inputs.to(dtype=torch.float64).cuda()
                    labels = labels.to(dtype=torch.float64).cuda()
                
                outputs = model(inputs)

                # Applying Flip test
                if flip_test:
                    outputs_1 = model(inputs.flip(dims=(3,)))
                if isinstance(outputs, list):
                    outputs = outputs[0]
                    if flip_test:
                        outputs_1 = outputs_1[0]
                
                #In case outputs contain a dict key
                if isinstance(outputs, dict):
                    if flip_test:
                        hm_outputs = (outputs['hm'] + outputs_1['hm'])/2
                        cls_outputs = (outputs['cls'] + outputs_1['cls'])/2
                    else:
                        hm_outputs = outputs['hm']
                        cls_outputs = outputs['cls']
                logger.info('Inferencing time --- {}'.format(time.time() - st))

                #For video_level or not, store by vid_id
                for idx, vid_id in enumerate(vid_ids):
                    if vid_id in vid_preds.keys(): 
                        vid_preds[vid_id] = torch.cat((vid_preds[vid_id], torch.unsqueeze(cls_outputs[idx], 0)), 0)
                    else:
                        vid_preds[vid_id] = torch.unsqueeze(cls_outputs[idx].clone().detach(), 0).cuda().to(dtype=torch.float64)
                        vid_labels[vid_id] = torch.unsqueeze(labels[idx].clone().detach(), 0).cuda().to(dtype=torch.float64)
            
            #Out of batch loop
            #Concatenate by video
            if video_level:
                for k in vid_preds.keys():
                    #Pred for a video is the mean of predictions on all frames (in logits)
                    total_preds = torch.cat((total_preds, torch.mean(vid_preds[k], 0, keepdim=True)), 0)
                    #Store labels (video name)
                    total_labels = torch.cat((total_labels, vid_labels[k]), 0)

            vid_probs = get_vid_probs(vid_preds)
            SBI = args.type.lower() == 'sbi'
            save_as_csv(vid_probs, cfg.DATASET.DATA.TEST.ROOT, cfg.DATASET.DATA.TYPE, SBI)
            
