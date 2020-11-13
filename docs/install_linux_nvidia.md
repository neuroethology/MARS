# MARS installation instructions

>**Environment: Linux**
>
>**GPU: NVIDIA**


## Setting up your GPU for TensorFlow
MARS currently uses TensorFlow version 1.15. You won't need to install TensorFlow yourself, as installation is handled by the conda environment below-- but you will need to install **CUDA version 10.0 or later** and **cuDNN version 7.4 or later** to allow TensorFlow to interact with your GPU. You will also want to make sure your NVIDIA graphic drivers are up to date.

>If you have other versions of CUDA or cuDNN installed and are unwilling to upgrade them, then you may want to install the [MARS Docker Environment](Docker_instructions.md) instead.

First, **install CUDA** by following the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), the core of which is to:
1) Go to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), and enter your platform information.
2) Run the Base Installer instructions provided (may require `sudo`).
3) Call `nvcc --version` from your terminal to confirm that CUDA has installed correctly. You may need to launch a new terminal for the command to be recognized.

 Next, **install cuDNN** by following the [NVIDIA cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), which requires you to:
 1) Go to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn), select "Download cuDNN", and sign up for a Developer account.
2) Go to Downloads on the developer site, and select cuDNN version 7.4 or later. **Make sure you select a cuDNN version that is compatible with your CUDA version!** To check your CUDA version, launch a terminal and type `nvcc --version`.

3) Follow install instructions from [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux) to add cuDNN to your CUDA directory.

## Setting up the MARS conda environment
1) If you don't already have it, **install** [miniconda](https://docs.conda.io/en/latest/miniconda.html) by opening a terminal and entering:

     ```
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     sh ./Miniconda3-latest-Linux-x86_64.sh
     ```
    then following the install instructions.

  2) **Check** your install by opening a new terminal and typing `which python`. This should return something like `~/miniconda3/bin/python`.

  3) **Clone** or download the contents of this GitHub repository. For this guide we'll assume you clone/download everything to `~/MARS`:
      ```
      cd ~
      git clone --recurse-submodules https://github.com/neuroethology/MARS
      ```
  >Note: the `--recurse-submodules` also ensures that the contents of the `neuroethology/Util` repository are also cloned when you clone MARS. `Util` contains code that is common to multiple projects on [Neuroethology](https://github.com/neuroethology). If you didn't include this tag in the initial clone, you can clone the `Util` repo later by calling `git submodule update --init --recursive` from within the MARS directory.

  4) **Build** the MARS conda environment by entering the following:
      ```
      cd ~/MARS
      conda env create -f MARS_environment_linux.yml
      ```
Once the build has finished, you can activate the MARS environment by calling `conda activate mars` (or `system activate mars`).


## Downloading the trained MARS models
Before you can run MARS, you need to download the trained neural networks and classifiers MARS uses for pose estimation and behavior classification.

Models can be downloaded from [https://data.caltech.edu/records/1655](https://data.caltech.edu/records/1655). After downloading, unzip the `models` folder into the `MARS/mars_v1_8` directory. The contents of `MARS/mars_v1_8/models` should now be three directories called `classifier`, `detection`, and `pose`.

Now you're ready to detect some behaviors!
