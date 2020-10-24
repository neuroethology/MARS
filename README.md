# The **M**ouse **A**ction **R**ecognition **S**ystem (**MARS**)

MARS is an end-to-end computational pipeline for **tracking**, **pose estimation**, and **behavior classification** in interacting laboratory mice.

MARS can be run either through a graphical interface or from the command line.

### Installation
MARS can be run on Linux, Windows, or MacOS. We strongly recommend running MARS on a computer with a GPU; MARS runs fastest on NVIDIA GPUs with at least 10Gb memory.

The easiest way to set up MARS is with a conda environment, which will handle the installation of all necessary Python modules. In cases where this is not possible, we also provide a Docker environment that can be used on Linux machines (see next section).

The following instructions cover GPU setup and creation of the MARS conda environment:

|Operating System ( + GPU) | Install Instructions |
|---|:---:|
|Linux + NVIDIA | [link](docs/install_linux_nvidia.md) |
|Windows + NVIDIA | [link](docs/install_windows_nvidia.md) |
|Windows + other | [link](docs/) - TODO |
|Mac | [link](docs/) - TODO |


#### Docker support
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved and error-prone process. Additionally, because Docker for Windows does not support GPU access, we do not recommend this approach for Windows computers. Step-by-step instructions to set up the MARS Docker can be found [here](docs/Docker_instructions.md).
