FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PIP_DEFAULT_TIMEOUT 200

RUN apt-get update
RUN apt-get install -y sudo 
RUN apt-get install -y unzip 
RUN apt-get install -y graphviz 
RUN apt-get install -y curl 
RUN apt-get install -y musescore3 
RUN apt-get install -y pkg-config 
RUN sudo apt update
RUN sudo apt-get install -y libhdf5-serial-dev 
RUN sudo apt-get install -y python3-pip 


RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install -r /app/requirements.txt

# Hack to get around tensorflow-io issue - https://github.com/tensorflow/io/issues/1755
RUN pip install tensorflow-io
RUN pip uninstall -y tensorflow-io

COPY /notebooks/. /app/notebooks
COPY /scripts/. /app/scripts

RUN sudo apt-get install lsof

ENV PYTHONPATH="${PYTHONPATH}:/app"