FROM ubuntu

# Chainer
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python-dev python-pip
RUN pip install numpy six
RUN pip install chainer

# Animeface Training
RUN apt-get install libopencv-dev python-opencv unzip -y
RUN mkdir /animeface
COPY . /animeface/

WORKDIR /animeface

CMD ["sh", "train.sh"]
