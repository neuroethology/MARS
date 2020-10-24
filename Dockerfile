FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

USER root
ENV SHELL /bin/bash
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y wget
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git cmake gcc libqt4-dev python3.6-dev libxml2-dev libxslt1-dev libqtwebkit-dev python-pip python-opencv python-tk 

RUN pip install pip --upgrade
COPY . /app
WORKDIR /app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
&& chmod +x ./Miniconda3-latest-Linux-x86_64.sh\
&& bash ./Miniconda3-latest-Linux-x86_64.sh -b\
&& rm -f ./Miniconda3-latest-Linux-x86_64.sh
RUN conda env create -f MARS_environment_linux.yml\
 && source activate mars\
 && pip install colour_demosaicing hmmlearn==0.2.2

WORKDIR /app/mars_v1_8