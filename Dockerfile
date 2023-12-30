FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y software-properties-common
RUN apt-add-repository universe && apt-get update

ENV SHELL /bin/bash

#RUN apt-get install -y python3.6
#RUN DEBIAN_FRONTEND=noninteractive apt -y install python3-opencv python3-pip

COPY . /app
WORKDIR /app

#RUN pip3 install nvidia-pyindex
#RUN pip3 install --user --upgrade nvidia-tensorrt
#RUN export PIP_EXTRA_INDEX_URL='https://pypi.nvidia.com'
#RUN pip3 install --upgrade pip
#RUN pip3 install Cmake
#RUN pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn
#RUN pip3 install --user --extra-index-url https://pypi.nvidia.com tensorrt_libs
#RUN pip3 install -r docker_requirements_cuda11.txt

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /app
