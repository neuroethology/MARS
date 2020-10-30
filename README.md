# The **M**ouse **A**ction **R**ecognition **S**ystem (**MARS**)

MARS is an end-to-end computational pipeline for **tracking**, **pose estimation**, and **behavior classification** in interacting laboratory mice. MARS version 1.8 can detect attack, mounting, and close investigation behaviors in a standard resident-intruder assay.
<div align=center>
<img src='https://github.com/neuroethology/MARS/blob/develop/docs/mars_demo.gif?raw=true'>
</div>

#### System requirements
MARS can be run on Linux, Windows, or MacOS. We strongly recommend running MARS on a computer with a GPU.

#### Data requirements
MARS v1.8 works on top-view videos featuring pairs of interacting mice, with a black resident mouse and a white intruder mouse. MARS can be run on unoperated mice, or on videos in which one mouse has been implanted with a **cable-attached device** such as a microendoscope or fiberphotometry/optogenetic fiber.

![Example video frames from the MARS training set](docs/sample_arenas.png)

MARS performs best on videos taken in a standard home cage, at roughly 30Hz, and either in color or grayscale. We recommend the recording setup described in [Hong et al 2015](https://www.pnas.org/content/112/38/E5351.short), minus the depth camera (front-view camera is optional, and is not currently use by MARS.)

## Installation
The easiest way to run MARS is with a conda environment, which will handle the installation of all necessary Python modules. In cases where this is not possible, we also provide a Docker environment that can be used on Linux machines (see below).

The following instructions cover GPU setup and creation of the MARS conda environment:

|Operating System + GPU | Install Instructions |
|---|:---:|
|Linux + NVIDIA | [link](docs/install_linux_nvidia.md) |
|Windows + NVIDIA | [link](docs/install_windows_nvidia.md) |
|Mac | [link](docs/) - TODO |


#### Docker support
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved process. Also note that because Docker for Windows does not support GPU access, the MARS Docker container is currently Linux-only. Step-by-step instructions to set up the MARS Docker can be found [here](docs/Docker_instructions.md).

## Running MARS
 MARS can be run either through a graphical interface or from the command line.
