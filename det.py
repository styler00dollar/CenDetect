from mmdet.apis import single_gpu_test
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model
from mmcv import Config
import mmcv

import numpy as np
import glob
from tqdm import tqdm
import torch
import cv2
import os

# dont print deprication warnings
import warnings
warnings.filterwarnings("ignore")

fp16_cfg = True
confidence = 0.6
config_path = "mask_rcnn_r50_fpn_2x_coco.py"
model_path = "epoch_3.pth"
folder_path = "input/"
output_path = "output"

# detect all files from input folder
files = glob.glob(folder_path + '/**/*.png', recursive=True)
files_jpg = glob.glob(folder_path + '/**/*.jpg', recursive=True)
files.extend(files_jpg)

# build the model
cfg = Config.fromfile(config_path)
model = build_detector(cfg.model)

# load state dict into model
x = torch.load(model_path)
model.load_state_dict(x['state_dict'])

for i in tqdm(files):
  image = cv2.imread(i, cv2.IMREAD_COLOR)
  input = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)/255

  if fp16_cfg == True:
      wrap_fp16_model(model)
      model.half()
      input = input.type(torch.HalfTensor)

  # move to device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  input = input.to(device)
  model.eval()

  img_metas = {}
  img_metas['filename'] = i
  img_metas['ori_shape'] = image.shape
  img_metas['img_shape'] = image.shape
  img_metas['scale_factor'] = np.array([1., 1., 1., 1.])

  with torch.no_grad():
  #with torch.inference_mode(): # torch 1.9
    result = model.simple_test(input, [img_metas], rescale=None)

  # creating mask from detection
  counter = 0
  mask = np.zeros((image.shape[0], image.shape[1])).astype(bool)

  # bar
  for f in result[0][0][0]:
    if f[4] > confidence:
      mask = mask | result[0][1][0][counter]
      counter += 1

  counter = 0
  # mosaic
  for f in result[0][0][1]:
    if f[4] > confidence:
      mask = mask | result[0][1][1][counter]
      counter += 1

  # only save the mask
  #cv2.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(i))[0] + ".png"), np.array(mask).astype(np.uint8)*255)
  
  # save with image
  image[mask]= [0, 255, 0]
  cv2.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(i))[0] + ".png"), np.array(image).astype(np.uint8))