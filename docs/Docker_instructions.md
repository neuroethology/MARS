## Step-by-step instructions for installing the MARS Docker
Installing MARS via Docker instead of conda will give MARS more protection from changes to your host machine, however it is a more involved process. Users should be familiar with the concept of Docker and Docker containers.

### Install [Docker](https://www.docker.com/)

Follow these instructions (for Ubuntu): [https://docs.docker.com/install/linux/docker-ce/ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu).

You can verify that Docker was installed correctly by calling the `hello-world` image:
    ```
    docker run hello-world
    ```
### Install `nvidia-docker-2`
This will allow MARS to access your GPU from within a docker container. Follow the installation instructions at [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

  Verify that `nvidia-docker` was installed properly by calling:
  ```
  docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
  ```
  output from this call should be the same as when you call `nvidia-smi` from terminal. If this doesn't work, you may need to repeat the Docker install process and specify the version of Docker in your install command, for instance:
  ```
  apt-get install docker-ce=18.06.2~ce~3-0~ubuntu
  ```
### Build the MARS-docker image
This only needs to be done once, unless you would like to update to a more recent version of MARS, or add new trained models to MARS.

To build the Docker image, open the terminal and `cd` into this repo, then type:
```
docker image build -t mars-docker .
```
(you may need to `sudo` this.)

### Start the MARS container
This also only needs to be done once per session, and uses the MARS-docker image to start a container (an instantiation of an image) within which you can run MARS. Enter the  command:
  ```
  nvidia-docker run -e GPU=0 -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /media:/media -p 8888:8888 -dit --name MARS mars-docker
  ```
  A breakdown of this command:
  - `nvidia-docker run mars-docker` starts a container from the mars-docker image you just built.
  - `-e GPU=0` tells nvidia-docker to give the container access to GPU 0 (change id of GPU if necessary; you can check your GPU id by calling `nvidia-smi`)
  - `-e DISPLAY=$DISPLAY` lets the nvidia-docker send display output to your monitor. You can omit this command if you don't want to use the MARS gui.
  - `-e QT_X11_NO_MITSHM=1` and `-v /tmp/.X11-unix:/tmp/.X11-unix` let the gui interface properly with X11 for display rendering- these are also unnecessary if you don't intend to use the gui.
  - `-v /media:/media` will allow your container to access the contents of the `/media` directory on the host machine; the `:/media` part means the contents of `/media` will appear in `/media` within the docker container (modify as desired). If you store your data somewhere else, you should modify this part to `-v /path/to/your/data:/path/within/docker`
  - `-p 8888:8888` sets the port so you can use tensorboard or jupyter within the container via port 8888 (modify port number as desired).
  - `-dit` starts the container in detached + interactive mode
  - `--name MARS` is the name of the container you're starting (can be anything)

  > **_NOTE:_** On some computers, the nvidia docker image MARS is based on fails to install cudnn5.1 correctly, without which tensorflow won't run on the GPU. If you have cudnn5.1 on your local machine, you can hackily fix this by adding the argument `-v /usr/local/cuda:/usr/local/cuda` to the command above, ie borrowing the needed files from the host computer. Another option is to enter the built container and install cudnn5.1 manually. Still not sure what is causing cudnn5.1 installation to fail.

  To confirm that your container is running, enter the command `docker ps -a` into terminal. You should see a container named "MARS" that was just created:
  ```
  CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                      PORTS               NAMES
  b065b587a340        mars-docker         "/bin/bash"         18 minutes ago      Exited (0) 16 seconds ago                       MARS
  ```

### Enter the MARS container and run MARS
Once the container has been created, enter it by calling `docker attach MARS`. You should see a command line that looks like: `root@b065b587a340:/app/MARS_v1_7#` (you may need to hit enter more than once for the prompt to show up).

If you get an error message saying the container has not been started, call `docker start MARS` then try again.

To test out MARS, call `python MARS_v1_7.py`. This should launch the MARS gui. If you're getting an error about being unable to access the display, enter the command `xhost local:root` in terminal, then try re-starting the container.


Hit the Browse button and navigate to the folder containing your videos, then press "Enqueue". Check boxes for the analyses you'd like to run, then click "Run MARS". You can track MARS's progress using the progress bars in the gui, or from the text output to the terminal.
