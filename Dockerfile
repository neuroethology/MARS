FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

USER root
ENV SHELL /bin/bash

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git cmake gcc libqt4-dev python2.7-dev libxml2-dev libxslt1-dev libqtwebkit-dev python-pip python-opencv python-tk 

RUN pip install pip --upgrade
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt --ignore-installed six

# run after the rest so pip can find other required packages:
# only need PySide for the gui component of MARS- otherwise keep it commented, it's very slow to install
RUN pip install hmmlearn==0.2.0 && pip install PySide

WORKDIR /app/MARS_v1_7
#CMD python MARS_v1_7.py