# Step-by-step instructions for installing the MARS Docker
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved process. Users should read up on the concept of Docker and Docker containers to understand what's happening.

>Unfortunately, Docker on Windows is unable to run tasks on the GPU, so we do not recommend Docker-based installation of MARS on Windows computers at this time.

## Install [Docker](https://www.docker.com/)

**Instructions for Linux:** [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).
<!---
**Instructions for MacOS:** [https://docs.docker.com/docker-for-mac/install/](https://docs.docker.com/docker-for-mac/install/).
**Instructions for Windows 10:** [https://docs.docker.com/docker-for-windows/install/](https://docs.docker.com/docker-for-windows/install/).
> Windows Security can sometimes prevent Docker from launching in Windows 10. If you have this issue, follow these steps:
>* Open "Window Security"
>* Open "App & Browser control"
>* Click "Exploit protection settings" at the bottom
>* Switch to "Program settings" tab
>* Select "Add program to customize" and navigate to `C:\WINDOWS\System32\vmcompute.exe`
>* Click "Edit"
>* Scroll down to "Code flow guard (CFG)" and uncheck "Override system settings"
>* Start vmcompute from Powershell with command `net start vmcompute`
-->

Verify that Docker was installed correctly by calling `docker run hello-world` from terminal (same command on all operating systems). This should output a "Hello from Docker!" message with some additional text.

## Install `nvidia-docker2`
This will allow MARS to access your GPU from within a docker container. Follow the installation instructions [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](here).

  Verify that `nvidia-docker2` was installed properly by calling:
  ```
  sudo docker run --rm --gpus all nvidia/cuda:10.0-base nvidia-smi
  ```
  output from this call should be the same as when you call `nvidia-smi` from terminal.
  >(Note: since the official `nvidia-docker2` install instructions all use `sudo`, we will be using `sudo` for all docker commands from here onward.)

## Build the MARS-docker image
This only needs to be done once, unless you would like to update to a more recent version of MARS, or add new trained models to MARS.

If you haven't already, download MARS from GitHub:
```
git clone --recurse-submodules https://github.com/neuroethology/MARS
```
To build the Docker image, `cd` into the MARS repository and type:
```
sudo docker image build -t mars-docker .
```

## Start the MARS container
This also only needs to be done once per session, and uses the MARS-docker image to start a container (an instantiation of an image) within which you can run MARS. Enter the  command:
  ```
  sudo docker run --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix-v /media:/media -p 8888:8888 -dit --name MARS mars-docker bash
  ```
  A breakdown of this command:
  - `docker run mars-docker` starts a container from the mars-docker image you just built.
  - `--gpus all` tells docker to give the container access to the computer's GPUs.
  - `-e DISPLAY=$DISPLAY` lets the nvidia-docker send display output to your monitor. You can omit this command if you don't want to use the MARS gui.
  - `-e QT_X11_NO_MITSHM=1` and `-v /tmp/.X11-unix:/tmp/.X11-unix` let the gui interface properly with X11 for display rendering- these are also unnecessary if you don't intend to use the gui.
  - `-v /media:/media` will allow your container to access the contents of the `/media` directory on the host machine; the `:/media` part means the contents of `/media` will appear in `/media` within the docker container (modify as desired). If you store your data somewhere else, you should modify this part to `-v /path/to/your/data:/path/within/docker`
  - `-p 8888:8888` sets the port so you can use tensorboard or jupyter within the container via port 8888 (modify port number as desired).
  - `-dit` starts the container in detached + interactive mode
  - `--name MARS` is the name of the container you're starting (can be anything)
  - `bash` means you'll connect to the bash terminal when you enter the image

  > **_NOTE:_** These instructions are for Docker version 19.03 and later. If you use an earlier version of Docker, you should instead enter the image using `nvidia-docker`, like so: `sudo nvidia-docker run -e GPU=0 -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /media:/media -p 8888:8888 -dit --name MARS mars-docker`

  To confirm that your container is running, enter the command `docker ps -a` into terminal. You should see a container named "MARS" that was just created:
  ```
  CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                      PORTS               NAMES
  b065b587a340        mars-docker         "/bin/bash"         18 minutes ago      Exited (0) 16 seconds ago                       MARS
  ```

### Enter the MARS container and launch MARS
Once the container has been created, enter it by calling `sudo docker attach MARS`. You should see a command line that looks like: `root@b065b587a340:/app/MARS_v1_8#` (you may need to hit enter more than once for the prompt to show up).

If you get an error message saying the container has not been started, call `sudo docker start MARS` then try again.

To test out MARS, call `python MARS_v1_8.py`. This should launch the MARS gui. If you're getting an error about being unable to access the display, enter the command `xhost local:root` in terminal, then try re-starting the container.
