FROM ubuntu

# Chainer
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python-dev python-pip
RUN pip install numpy six
RUN pip install chainer

# Animeface Training
RUN apt-get install libopencv-dev python-opencv git unzip -y
