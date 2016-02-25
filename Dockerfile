FROM ubuntu

# chainer
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python-dev python-pip
RUN pip install numpy six
RUN pip install chainer

# animeface training
RUN apt-get install imagemagick libopencv-dev python-opencv -y
