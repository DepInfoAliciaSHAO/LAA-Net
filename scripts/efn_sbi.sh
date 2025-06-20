#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py --cfg configs/efn4_fpn_sbi_adv.yaml
