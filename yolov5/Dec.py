"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import os
sys.path.append(os.getcwd()+r'\\yolov5')
from pathlib import Path

import cv2
import torch
import numpy as np


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box


@torch.no_grad()
def run(
        model,
        weights='',  # model.pt path(s)
        img ='',
        conf_thres='',  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        ):

    # Inference
    pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    pred = pred[0].cpu().tolist()

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    return pred


def parse_opt(conf,img0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--img', default=img0, help='输入的图像')
    parser.add_argument('--conf-thres', type=float, default=conf, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    return opt


def yoloDect(img0,model,conf):
    opt = parse_opt(conf,img0)
    result = run(model, **vars(opt))
    return np.array(result)

