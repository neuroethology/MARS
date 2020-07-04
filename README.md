# The **M**ouse **A**ction **R**ecognition **S**ystem (**MARS**)

MARS is an end-to-end computational pipeline for **tracking**, **pose estimation**, and **behavior classification** in interacting laboratory mice.

MARS can be run either through a graphical interface or from the command line.

### Installation
Installation of MARS can be managed via either a conda environment or a Docker container. To run MARS you will need a Linux computer with an Nvidia GPU.

#### Step-by-step instructions for installing the MARS conda environment
1) If not already installed, install [miniconda](https://docs.conda.io/en/latest/miniconda.html) by opening a terminal and entering:

     ```
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     sh ./Miniconda3-latest-Linux-x86_64.sh
     ```
    then following the install instructions. Check your install by opening a new terminal and typing `which python`. This should return something like `~/miniconda3/bin/python`.
<br>
2) Clone or download the contents of this GitHub repository.
<br>
3) Build the MARS conda environment by navigating to your local copy of the MARS GitHub repository and calling
    ```
    conda env create -f MARS_environment.yml
    ```
    Once the build has finished, you can activate the MARS environment by calling `conda activate mars_tf` (on some systems instead use `system activate mars_tf`).


#### Step-by-step instructions for installing the MARS Docker
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved process. Step-by-step instructions can be found [here](README_docker.md).
