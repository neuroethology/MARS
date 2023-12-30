FROM tensorflow/tensorflow:1.15.0-gpu-py3

ENV SHELL /bin/bash

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt -y install python3-opencv

COPY . /app
WORKDIR /app

RUN pip3 install -r docker_requirements.txt

WORKDIR /app/mars_v1_8
