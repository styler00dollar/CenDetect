# CenDetect

A repository to detect degradation in images and masking such areas.

**Warning: Do not share modified images, since randomly grabbed images are used for dataset purposes. If I notice a growing amount of bad images in my dataset, I will probably stop sharing anything.**

# Features
- No Tensorflow (easy to install, no GPU restrictions by forcing certain versions of CUDA, no very specific env requirements)
- FP16 (higher speed on RTX2xxx or newer GPUs and maybe less VRAM usage)
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
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# or installing with Anaconda
conda create -n mmdetection python=3.9 -y
conda activate mmdetection
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# CPU
# Installing PyTorch with pip
pip install torch==1.9.0+cpu torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# or with Anaconda (currently does show errors inside anaconda with 3.9, but 3.8 seems to work fine)
conda create -n mmdetection python=3.8 -y
conda activate mmdetection
conda install pytorchn=1.9.0 torchvision torchaudio cpuonly -c pytorch -y
# if you really want 3.9, install it with CUDA. That is bigger in total size, but can also run cpu
conda install pytorch=1.9.0 torchvision -c pytorch -y

# Alternatively you can visit https://pytorch.org/get-started/locally/ and look up install commands

# Install other dependencies (pip also works in Anaconda)
pip install opencv-python tqdm numpy
pip install timm --no-deps

# Installing pre-compiled mmcv
pip install mmcv-full

# Compiling mmcv instead
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .

# Installing mmdetection
git clone https://github.com/styler00dollar/Colab-mmdetection mmdetection
cd mmdetection
pip install -e .
```

## Windows
```
# Install PyTorch, numpy, tqdm, timm and OpenCV with the commands above
# Download Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
# Select the first thing with C++ inside the installer and install
# Reboot
# Download and install Cuda 11.1 (https://developer.nvidia.com/cuda-11.1.0-download-archive)
# Reboot

# If Anaconda is used, activate env
conda activate mmdetection
# Installing with pip is not recommended inside of Windows, since it can result in DLL errors, compile it
# if you do not use Anaconda, download the Code from the mmcv Github manually instead of using git commands
conda install git -y
git clone https://github.com/open-mmlab/mmcv.git

cd mmcv
set MMCV_WITH_OPS=1
pip install -e .

# if you still want to try to install with pip, here is the command
pip install mmcv-full

# Same thing for mmdetection, either use git with conda or download Code manually and cd into it
git clone https://github.com/styler00dollar/Colab-mmdetection mmdetection
cd mmdetection
pip install -e .
```

## Compiling PyTorch
If you really want to compile PyTorch instead of using the pre-compiled version in the instructions above, then use these steps. One of the main reasons for doing so is that there currently is no Cuda 11.3/11.4 binary for PyTorch, but it can be compiled.

Linux
```
# if you use conda
conda activate mmdetection

git clone https://github.com/pytorch/pytorch
cd pytorch
# example selection of branch
git checkout remotes/origin/release/1.9

git submodule update --init --recursive

# install directly
python setup.py install
or
pip install -e .

# or create whl to install
python setup.py bdist_wheel
# find the created .whl file and cd into it
pip install package_name.whl
```
Windows (Instructions for conda, since git is needed)
```
# Download Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
# Select the first thing with C++ inside the installer and install
# Reboot
# Download and install Cuda (https://developer.nvidia.com/cuda-downloads)
# Reboot

conda activate mmdetection
conda install git -y

git clone https://github.com/pytorch/pytorch
cd pytorch
# example selection of branch
git checkout remotes/origin/release/1.9

pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing ninja typing_extensions
python setup.py bdist_wheel --cmake

# find the created .whl file and cd into it
pip install package_name.whl
```

## Usage
```
# If Anaconda is used, make sure you use the correct env
conda activate mmdetection

python det.py [--fp16] [--confidence <float (0-1)>] [--input_path "PATH"] [--output_path "PATH">] [--device "cuda" or "cpu"] [--model "mask_rcnn_r50_fpn" | "mask_rcnn_r101_fpn" | "point_rend_r50_fpn" | "mask_rcnn_r50_fpn_dconv"]
```

There are currently 4 models. Trying to figure out what would be better or worse.
|    Model     |  CPU  | GPU (CUDA) | FP16 | Iterations / Batch Size / MinMax Train Res | Speed CPU | Speed GPU (FP16/FP32) | VRAM Usage (FP16/FP32) | bbox mAP / mAR @IoU=0.50:0.95 |  segm mAP / mAR @IoU=0.50:0.95 
| :----------: | :---: | :--------: | :--: | :----------------------------------------: | :-------------------: | :-------------------: | :--------------------: | :------------: | :---: | 
|   mask_rcnn_r50_fpn       | Yes | Yes | Yes | 235k / 2 / 1024-1666px | 7.24s | 0.158s / 0.185s | 2.7GB / 2.6GB | 0.757 / 0.819 | 0.808 / 0.855
|   mask_rcnn_r101_fpn      | Yes | Yes | Yes | 190k / 2 / 1024-1500px | 9.13s | 0.165s / 0.2s | 2.3GB / 2.2GB | 0.756 / 0.811 | 0.792 / 0.840
|   point_rend_r50_fpn      | Yes | Yes (mmcv + pytorch 1.9 seems stable) | No (grid_sampler wants FP32) | 520k / 2 / 1024-1600px | 6.88s | x / 0.192s | x / 2.4GB | 0.735 / 0.787 | 0.803 / 0.846 
|   cascade_mask_rcnn_r50_fpn_dconv | No (DeformConv is not implemented on CPU) | Yes | Yes | 125k / 2 / 1024-1600px | | 0.174 / 0.194 | 2.3GB / 2.4GB | 0.787 / 0.838 | 0.764 / 0.813 |

FP16 can be faster (especially if the GPU is RTX2xxx or newer) and *can* use less VRAM, but tests did not really show VRAM improvements. **Warning: Do not use FP16 on hardware that does not have Tensor Cores!** Due to overhead because of converting data into FP16, it can also be slower. Tests with a P100 showed slower inference speeds compared to FP32. Only use it on GPUs that are RTX2xxx and newer.

VRAM usage and speed tests use 174 1024px resized png files with a Tesla V100, which does have Tensor Cores. CPU tests are done on a Intel Xeon Dual-Core CPU @ 2.20GHz. VRAM measurements seem to vary a bit, use it as a rough estimate. The eval dataset was filtered with ``cv2.contourArea(border_contours) > 7`` to avoid crashing. Afterwards I noticed that the eval size is the same as training size, it is not the same for every model. Maybe I will re-run everything to make it more fair.

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

# optional argument to make one .exe
--onefile
```

# How to manually annotate data (Linux)
The code works with coco json files. There are probably several tools to do so, I tested `coco-annotator` with arch. Use the labels "bar" and "mosaic".

Warning: Make sure that every single mask has a seperate bounding box. Create a new detection for every mask with "+". If you don't do that, it will be technically one big mask (one object, instead of multiple), which is not good for training. In `coco-annotator` you should see different colors for every mask inside the same picture. Also, try to mask it as good as possible. Bad data that is a lot of pixels off is not useful.

Also important, delete all duplicates with some tool (for example wiht [dupeguru](https://github.com/arsenetar/dupeguru)) and hash your files with MD5. [I have a simple tool for this.](https://github.com/styler00dollar/MD5Renamer)
```
# Just put this in the folder with pictures and run it
wine MD5RenamerConsole64.exe .
```
Example for several mosaic masks:
<p align="center">
  <img src="https://user-images.githubusercontent.com/51405565/125399990-9b59c400-e3b1-11eb-9a1b-a288196cb676.png" />
</p>

Install instructions for `coco-annotator`
```
yay -S docker
yay -S docker-compose
git clone https://github.com/jsbroks/coco-annotator
```
How to start it (will install in the first startup)
```
cd coco-annotator
# fixing docker errors
systemctl start docker
sudo chmod 666 /var/run/docker.sock
# start docker
docker-compose up

# visit this url, does not matter what account details you use, just create some account
# it will open a page in your browser where you can annotate files and export them into json afterwards
http://localhost:5000/

# you can turn it off again with
docker-compose down
# it does save automatically, you can resume at any point again
```

If you want to merge data with mine, I am `sudo rm -rf / --no-preserve-root#8353` in discord.

# How to manually annotate data (Windows)
Hash the files like in the linux method, but just use ``MD5RenamerConsole64.exe .`` instead. Also make sure your labels look like in the linux screenshot.

```
# Download Docker (https://docs.docker.com/docker-for-windows/install/)
# Install as Admin!
# Reboot
# Run as Admin!

# Open powershell
git clone https://github.com/jsbroks/coco-annotator
cd coco-annotator
# start
docker-compose up

# visit this url, does not matter what account details you use, just create some account
# it will open a page in your browser where you can annotate files and export them into json afterwards
http://localhost:5000/
```

# How to manually annotate data (png files)

An alternative is to have mask/file pairs, from which json files can be created. Hash the files like in the Linux method. A suggestion would be do do:
```
/masks
- b40f4afe1ce30dfd69f6fc6308531ed0.png (black image with [0,255,0] for bar and [0,255,255] for mosaic)
...

/data
- b40f4afe1ce30dfd69f6fc6308531ed0.png
...
```
The exact format is not that important. The only important thing is that a mask has a unique color and classes can be distinguished with filenames, so the mask data can be extracted and can not be confused with image data.

# Future plans
Improving the dataset to some degree, already trying. I don't think other models could improve the detection itself significantly. Maybe different backbones could, but they usually want a lot of VRAM.

# FAQ
Q: AMD GPU?

A: No, you bought a bad GPU. Only Nvidia GPUs do have good support for this kind of task. Code will use CPU instead. It may be possible with Vulkan or ONNX, but this requires some work and is most likely also slower, [since Vulkan showed around half of the performance of CUDA](https://github.com/n00mkrad/flowframes/blob/main/Benchmarks.md). Exporting ``mask_rcnn_r50_fpn_2x_coco`` for example results in ``failed:Fatal error: grid_sampler is not a registered function/op``. Simply exporting a model to ONNX most likely won't work.

If you *really* want to try to run the source Code and have a GPU that is ROCm compatible, you can try to get the ROCm version of PyTorch working. I only got ROCm working in Arch personally. If you have a RX4xxx GPU, then you need to compile it yourself. Install Arch linux and [follow these steps](https://github.com/pytorch/pytorch/issues/53738#issuecomment-813058293). Compiling ROCm and Pytorch (ROCm version) will probably take like 9 hours in total. Around 6 for ROCm and around 3 for Pytorch. No gurantee it will even work, since my attempt to get EfficientNet working failed. Binaries should work if you have a RX5xx or a Vega and install instructions are similar for these GPUs if you want to compile it. Installing mmcv-full will fail because CUDA is missing, but maybe it could work if you install the lite version without custom CUDA ops. Such an approach would assume that the models don't use these operations and it won't work for every model. Maybe a ROCm version of mmcv will be relased soon [according to this](https://github.com/open-mmlab/mmcv/pull/1022).

Q: Supported file formats?

A: Technically everything that OpenCV supports, but glob currently only searches for JPG, JPEG, WebP and PNG.

# Acknowledgements
Was trained with the [OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection). Mostly unmodified, but custom code was written to use the trained models. [Also custom timm backbones and new optimizers were added in my fork](https://github.com/styler00dollar/Colab-mmdetection).

Inspriration from [natethegreate/hent-AI](https://github.com/natethegreate/hent-AI). Mostly giving ideas on which custom code was written.
