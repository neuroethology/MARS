# The **M**ouse **A**ction **R**ecognition **S**ystem (**MARS**)

MARS is an end-to-end computational pipeline for **tracking**, **pose estimation**, and **behavior classification** in interacting laboratory mice. MARS version 1.8 can detect attack, mounting, and close investigation behaviors in a standard resident-intruder assay.

### Quick Navigation
* [Requirements](#requirements)
    * [System requirements](#system-requirements)
    * [Data requirements](#data-requirements)
* [Installation](#installation)
    * [Docker support](#docker-support)
* [Running MARS](#running-mars)
    * [Running from the Python terminal](#running-from-the-python-terminal)
    * [Running from the MARS GUI](#running-from-the-mars-gui)
 * [Troubleshooting](#troubleshooting)

<div align=center>
<img src='https://github.com/neuroethology/MARS/blob/develop/docs/mars_demo.gif?raw=true'>
</div>

## Requirements
### System requirements
MARS can be run on Linux, Windows, or MacOS. We strongly recommend running MARS on a computer with a GPU.

### Data requirements
MARS v1.8 works on top-view videos featuring pairs of interacting mice, with a black resident mouse and a white intruder mouse. MARS can be run on unoperated mice, or on videos in which one mouse has been implanted with a cable-attached device such as a microendoscope or fiberphotometry/optogenetic fiber.

![Example video frames from the MARS training set](docs/sample_arenas.png)

MARS performs best on videos taken in a standard home cage, at roughly 30Hz, and either in color or grayscale. We recommend the recording setup described in [Hong et al 2015](https://www.pnas.org/content/112/38/E5351.short), minus the depth camera (front-view camera is optional, and is not currently used in pose estimation.)

## Installation
The easiest way to run MARS is with a conda environment, which will handle the installation of all necessary Python modules. In cases where this is not possible, we also provide a Docker environment that can be used on Linux machines (see below).

The following instructions cover GPU setup and creation of the MARS conda environment:

|Operating System + GPU | Install Instructions |
|---|:---:|
|Linux + NVIDIA | [link](docs/install_linux_nvidia.md) |
|Windows + NVIDIA | [link](docs/install_windows_nvidia.md) |
|Mac | (coming soon) |


### Docker support
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved process. Also note that because Docker for Windows does not support GPU access, the MARS Docker container is currently Linux-only. [Instructions to set up the MARS Docker can be found here.](docs/Docker_instructions.md)

## Running MARS
 MARS can be run either through a graphical interface or from Juypter/Python terminal. Before running MARS, make sure you do the following:
 - Add the `mars_v1_8` directory to your Python search path (or, run Python from within this directory.)
 - Activate the MARS conda environment by calling `conda activate mars` or `source activate mars`.

### Running from the Python terminal
To run MARS with default settings, simply call the following from within Python:
```
import MARS
folder = ['\path\to\your\videos']
MARS.run_MARS(folder)
```
> **Important note**: for MARS to find your movie, **please give it a file name ending in "_Top"** (indicating that it was filmed from a top-view camera.) MARS currently runs on videos in `*.seq`, `*.avi`, `*.mp4`, or `*.mpg` format.

MARS default settings can be found within `config.yml`, along with descriptions for what each setting does. You can change settings either by modifying this file, or by passing a second "options" parameter in your call to `run_MARS`. For example, here we turn off behavior classification, and tell MARS to overwrite any existing pose/feature files:
```
opts = {'doActions': False, 'doOverwrite': True}
MARS.run_MARS(folder, user_opts=opts)
```
Please refer to `config.yml` to see the full list of available options.

### Running with the MARS GUI
To run MARS using the GUI, take the following steps:
1. Depending on your system:
   - **Windows**:  launch an Anaconda Prompt by searching Anaconda in the Start Menu.
   - **Linux**:  launch a new terminal.
2. Navigate to the MARS v1.8 directory by calling `cd Documents/Github/MARS/mars_v1_8` (or whatever the path to MARS is on your system.)
3. Activate the MARS conda environment by typing `conda activate mars` (or `system activate mars`).
4. Launch the MARS gui by calling:
```
python MARS.py
```
To select videos to analyze, click <kbd>Browse</kbd> and navigate to the directory containing your videos. **Note**: the current version of MARS will only find your videos if they have filenames ending in "_Top"!

If MARS finds any videos in the selected directory, it will display additional buttons to set run options:
<div align=center>
<img src='https://github.com/neuroethology/MARS/blob/develop/docs/mars_gui.png?raw=true'>
</div>
Click <kbd>Enqueue</kbd> to add your selected folder to MARS's work queue, then select the camera view you use (typically "Top") and the tasks you want to run (typically "Pose", "Features", and "Classify Actions".) Check "Produce Video" if you'd like MARS to generate a video of the tracked poses, and check "Overwrite" to overwrite previously generated MARS output. Finally, click the green <kbd>Run MARS</kbd> button to start running!

You can track MARS's progress using the progress bars in the gui, or refer to your terminal for more detailed progress updates.

## Troubleshooting
If you encounter any errors while running MARS, take the following steps:
1. Check the [issues page](https://github.com/neuroethology/MARS/issues?q=is%3Aissue+) to see if anyone has encountered the same issue.
2. If you can't resolve it, either submit a **new issue** on the issues page, or contact us <A HREF="&#109;&#97;&#105;&#108;&#116;&#111;&#58;%61%6E%6E%2E%6B%65%6E%6E%65%64%79%40%6E%6F%72%74%68%77%65%73%74%65%72%6E%2E%65%64%75">via email</A>.
