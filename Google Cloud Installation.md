# Google Cloud Installation

## Remove GPU drivers
```bash
sudo apt-get purge nvidia*
sudo apt-get purge cuda*
```

## Reinstall GPU driver
```bash
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install nvidia-384-dev
sudo reboot
```

## check driver setting
```bash
lsmod | grep nvidia
nvidia-smi
```

## Install CUDA 9 and CuDNN 7 
based on [this link](http://goodtogreate.tistory.com/entry/TensorFlow-GPU-버전-우분투-1604에-설치-하기)

```bash
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-9.0

# Copy libcudnn7* files from nvidia
cd /tmp
sudo dpkg -i libcudnn7_*cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev*cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc*cuda9.0_amd64.deb
```

## Check driver setting again
```
sudo reboot
nvidia-smi
```

## Install tensorflow 1.4 
```
source ~/intQA/envIntQA/bin/activate
pip install tensorflow-gpu==1.4
```

## upgrade for for CUDA 9 and CuDNN 7 per https://github.com/mind/wheels/releases/tag/tf1.4-gpu-cuda9

```bash
pip install --ignore-installed --upgrade https://github.com/mind/wheels/releases/download/tf1.4-gpu-cuda9/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
```

## install intel-mkl 
per https://github.com/mind/wheels/releases/tag/tf1.4-gpu-cuda9
cd ~/intQA/ext

```bash
sudo apt install cmake
git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn/scripts && ./prepare_mkl.sh && cd ..
mkdir -p build && cd build && cmake .. && make
sudo make install
```