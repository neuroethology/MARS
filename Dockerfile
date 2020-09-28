FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

USER root
ENV SHELL /bin/bash

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y wget
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git cmake gcc libqt4-dev python3.6-dev libxml2-dev libxslt1-dev libqtwebkit-dev python-pip python-opencv python-tk 

RUN pip install pip --upgrade
COPY . /app
WORKDIR /app

# install required python packages. switched to doing this with conda.
#RUN pip install -r requirements_py3.txt --ignore-installed six
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sh ./Miniconda3-latest-Linux-x86_64.sh
RUN conda env create -f MARS_environment.yml

# run after the rest so pip can find other required packages:
# only need PySide for the gui component of MARS- otherwise keep it commented, it's very slow to install
# RUN pip install hmmlearn==0.2.0 # && pip install PySide

WORKDIR /app/mars_v1_8