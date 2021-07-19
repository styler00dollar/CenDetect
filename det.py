import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmcv.runner import wrap_fp16_model

import cv2
import numpy as np
from tqdm import tqdm
import os
import glob
import torch

# dont print deprication warnings
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--fp16', action='store_true', required=False, help='Enables FP16')
    parser.add_argument('--input_path', type=str, default="input", required=False, help='Path to a folder for input')
    parser.add_argument('--output_path', type=str, default="output", required=False, help='Path to a folder for output')


    # manually selecting config
    #parser.add_argument('--config_path', type=str, default="mask_rcnn_r50_fpn_2x_coco.py", required=False, help='Config file to build the model')
    #parser.add_argument('--model_path', type=str, default="epoch_3.pth", required=False, help='The path to the pth model itself')

    parser.add_argument('--model_directory', type=str, default="models", required=False, help='Folder path to configs and models')
    parser.add_argument('--model', type=str, default="mask_rcnn_r50_fpn", required=False, help='Seleting a model. [mask_rcnn_r50_fpn, mask_rcnn_r101_fpn, point_rend_r50_fpn, cascade_mask_rcnn_r50_fpn_dconv]')

    parser.add_argument('--device', default=None, help='Device used for inference')
    parser.add_argument('--confidence', type=float, default=0.3, required=False, help='Confidence thresh for detections (Values between 0 and 1)')
    args = parser.parse_args()
    return args

def main(args):
    # manually select model and config path
    #model = init_detector(args.config_path, args.model_path, device=args.device)

    # selection
    config_path = os.path.join(args.model_directory, args.model + ".py")
    model_path = os.path.join(args.model_directory, args.model + ".pth")

    if args.device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    model = init_detector(config_path, model_path, device=device)

    # detect all files from input folder
    files = glob.glob(args.input_path + '/**/*.png', recursive=True)
    files_jpg = glob.glob(args.input_path + '/**/*.jpg', recursive=True)
    files_jpeg = glob.glob(args.input_path + '/**/*.jpeg', recursive=True)
    files_webp = glob.glob(args.input_path + '/**/*.webp', recursive=True)
    files.extend(files_jpg)
    files.extend(files_jpeg)
    files.extend(files_webp)

    if args.fp16 == True:
        wrap_fp16_model(model)
        model.half()

    # test a single image
    for i in tqdm(files):
      image = cv2.imread(i, cv2.IMREAD_COLOR)

      result = inference_detector(model, i)

      # creating mask from detection
      counter = 0
      mask = np.zeros((image.shape[0], image.shape[1])).astype(bool)

      # bar
      for f in result[0][0]:
        if f[4] > args.confidence:
          mask = mask | result[1][0][counter]
          counter += 1

      counter = 0
      # mosaic
      for f in result[0][1]:
        if f[4] > args.confidence:
          mask = mask | result[1][1][counter]
          counter += 1

      # only save the mask
      #cv2.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(i))[0] + ".png"), np.array(mask).astype(np.uint8)*255)

      # save with image
      image[mask]= [0, 255, 0]
      cv2.imwrite(os.path.join(args.output_path, os.path.splitext(os.path.basename(i))[0] + ".png"), np.array(image).astype(np.uint8))


if __name__ == '__main__':
    args = parse_args()
    main(args)
