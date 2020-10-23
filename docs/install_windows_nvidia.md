### MARS installation instructions

>**Environment: Windows**
>**GPU: NVIDIA**


#### Setting up your GPU for TensorFlow
MARS currently runs on TensorFlow version 1.15. You won't need to install TensorFlow yourself, as installation is handled by the conda environment below-- but you will need to install **CUDA version 10.0 or later** and **cuDNN version 7.4 or later** to allow TensorFlow to interact with your GPU. You will also want to make sure your NVIDIA graphic drivers are up to date.

First, **install CUDA** by following the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/), the core of which is to:
1) Go to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), and enter your platform information.
2) Download and run the provided `*.exe` file.
3) Call `nvcc --version` from your terminal to confirm that CUDA has installed correctly.


 Next, **install cuDNN** by following the [NVIDIA cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), the core of which is to:
 1) Go to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn), select "Download cuDNN", and sign into/sign up for a Developer account.
2) In the "Downloads" menu on the developer site, select cuDNN version 7.4 or later. **Make sure you select a cuDNN version that is compatible with your CUDA version!**
  >To check your CUDA version, launch a terminal and type `nvcc --version`. If you just installed CUDA you may need to launch a new terminal for the command to be recognized.
3) Follow instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows) to add cuDNN to your CUDA directory.


#### Setting up the MARS conda environment
1) If you don't already have it, **install** miniconda by downloading the Python 3.x version appropriate to your system from [this page](https://docs.conda.io/en/latest/miniconda.html).
2) **Check** your install by opening a new **Anaconda Prompt** and typing `where python`. This should return one or more paths to python executables- make sure this includes something with `miniconda3`, like `C:\Users\me\miniconda3\python.exe`.  

3) **Clone** or download the contents of this GitHub repository. For this guide we'll assume you downloaded everything to `C:\users\me\MARS`.
    ```
    cd C:\users\me
    git clone --recurse-submodules https://github.com/neuroethology/MARS
    ```
    (or read about other ways to clone/download repos [here](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
>Note: the `--recurse-submodules` also ensures that the contents of the `neuroethology/Util` repository are also cloned when you clone MARS. `Util` contains code that is common to multiple projects on [Neuroethology](https://github.com/neuroethology). If you didn't include this tag in the initial clone, you can clone the `Util` repo later by calling `git submodule update --init --recursive` from within the MARS directory.

4) **Build** the MARS conda environment by entering the following in an Anaconda Prompt:
    ```
    cd C:\users\me\MARS
    conda env create -f MARS_environment_windows.yml
    ```
  Once the build has finished, you can activate the MARS environment by calling `conda activate mars` (or `system activate mars`).
