# CenDetect

A repository to detect degradation in images and masking such areas.

**Warning: Do not share modified images, since randomly grabbed images are used for dataset purposes. If I notice a growing amount of bad images in my dataset, I will probably stop sharing anything.**

# Features
- No Tensorflow (easy to install, no GPU restrictions by forcing certain versions of CUDA, no very specific env requirements)
- FP16 (less VRAM usage and potentially higher speed)
- Uses an Nvidia GPU if there is one available, if not, then CPU
- Big dataset with augmentations
- Multiple architectures (since it uses mmdetection as a base)

# Running the compiled version (Windows)
Just download from ``releases`` and open the ``.bat`` file. You can also edit that file to change the startup parameters, like ``confidence`` and ``fp16``.

# Running from source
There are 2 ways to install it, with and without Anaconda. If you don't want to use Anaconda, install normal Python. Python is already preinstalled in Linux based distros. Anaconda is recommended. If you do not use Anaconda, then you maybe need to install CUDA in Linux yourself. Inside Windows, it seems to be a requirement to install CUDA.

The exact version of Python and the other packages shouldn't be very important, but make sure that [mmcv](https://github.com/open-mmlab/mmcv) does support the installed PyTorch package. Currently it ``supports up to Pytorch 1.8``. ``Pytorch 1.9 currently is not compatible`` and will result in errors.

Anaconda: https://www.anaconda.com/products/individual

Python: https://www.python.org/downloads/

## Linux
```
# GPU
# Installing PyTorch with pip
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# or installing with Anaconda
conda create -n mmdetection python=3.9 -y
conda activate mmdetection
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# CPU
# Installing PyTorch with pip
pip install torch==1.8.0+cpu torchvision==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# or installing with Anaconda
conda create -n mmdetection python=3.9 -y
conda activate mmdetection
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y

# Install other dependencies (pip also works in Anaconda)
pip install opencv-python tqdm numpy

# Installing mmcv with CUDA 11.1 (GPU usage)
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
# Installing mmcv CPU version
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

# Installing mmdetection
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip install -e .
```

## Windows
```
# Install PyTorch, numpy, tqdm, OpenCV and mmcv with the commands above
# Download and install Cuda 11.1 (https://developer.nvidia.com/cuda-11.1.0-download-archive)
# Reboot
# Download Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
# Select the first thing with C++ inside the installer and install
# Reboot

# If Anaconda is used, activate env
conda activate mmdetection

pip install mmcv-full
# Download the official mmdetection Code (https://github.com/open-mmlab/mmdetection)
# Change `mmdetection/requirements/runtime.txt` file to `pycocotools` instead of `pycocotools; platform_system == "Linux"` and `pycocotools-windows; platform_system == "Windows"`.
# If you don't change this, it will say that it couldn't find the package "pycocotools"
# run install within the mmdetection directory
pip install -e .
```

## Usage
```
# If Anaconda is used, make sure you use the correct env
conda activate mmdetection

python det.py [--fp16] [--confidence <float (0-1)>] [--input_path "PATH"] [--output_path "PATH">] [--device "cuda" or "cpu"] [--model "mask_rcnn_r50_fpn" | "mask_rcnn_r101_fpn" | "point_rend_r50_fpn" | "mask_rcnn_r50_fpn_dconv"]
```

There are currently 4 models:
|    Model     |  CPU  | GPU (CUDA) | FP16 | Training iter | Misc
| :----------: | :---: | :--------: | :--: | :-----------: | :--: |
|   mask_rcnn_r50_fpn       | Yes | Yes | Yes | ~172k | Fast with medium accuracy
|   mask_rcnn_r101_fpn      | Yes | Yes | Yes | X | A bigger version of mask_rcnn_r50_fpn, theoretically better
|   mask_rcnn_r50_fpn_dconv | No  | Yes | Yes | X | Should be better than mask_rcnn_r50_fpn
|   point_rend_r50_fpn      | Yes | No (1.8.1+cu111 bug, waiting for 1.9) | No | ~433k | Should be better than mask_rcnn_r50_fpn_dconv


FP16 can be faster (especially if the GPU is RTX2xxx or newer) and uses less VRAM.

# Creating an .exe with pyinstaller
```
# Do the steps for Windows above
# If conda is used, uninstall conda Pillow to avoid problems
conda uninstall Pillow -y
# or uninstall with pip
pip uninstall Pillow
pip install --upgrade Pillow

pip install pyinstaller
pyinstaller --hidden-import=mmcv._ext --hidden-import torchvision det.py
```

# FAQ
Q: AMD GPU?

A: No, you bought a bad GPU. Only Nvidia GPUs do have good support for this kind of task. Code will use CPU instead. It may be possible with Vulkan or ONNX, but this requires some work and is most likely also slower, [since Vulkan showed around half of the performance of CUDA](https://github.com/n00mkrad/flowframes/blob/main/Benchmarks.md). Exporting ``mask_rcnn_r50_fpn_2x_coco`` for example results in ``failed:Fatal error: grid_sampler is not a registered function/op``. Simply exporting a model to ONNX most likely won't work.

If you *really* want to try to run the source Code and have a GPU that is ROCm compatible, you can try to get the ROCm version of PyTorch working. I only got ROCm working in Arch personally. If you have a RX4xxx GPU, then you need to compile it yourself. Install Arch linux and [follow these steps](https://github.com/pytorch/pytorch/issues/53738#issuecomment-813058293). Compiling ROCm and Pytorch (ROCm version) will probably take like 9 hours in total. Around 6 for ROCm and around 3 for Pytorch. No gurantee it will even work, since my attempt to get EfficientNet working failed. Binaries should work if you have a RX5xx or a Vega and install instructions are similar for these GPUs if you want to compile it. Installing mmcv-full will fail because CUDA is missing, but maybe it could work if you install the lite version without custom CUDA ops. Such an approach would assume that the models don't use these operations and it won't work for every model. Maybe a ROCm version of mmcv will be relased soon [according to this](https://github.com/open-mmlab/mmcv/pull/1022).

Q: Supported file formats?

A: Technically everything that OpenCV supports, but glob currently only searches for JPG and PNG.

# Acknowledgements
Was trained with the [OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection). Mostly unmodified, but custom code was written to use the trained models and avoiding the usage of `.json` files, adding a replacement for `img_metas` and a `fp16` feature. [Also custom backbones where added with timm in my fork](https://github.com/styler00dollar/Colab-mmdetection).

Inspriration from [natethegreate/hent-AI](https://github.com/natethegreate/hent-AI). Mostly giving ideas on which custom code was written.
