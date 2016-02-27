# coding: utf-8

import os

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import cv2 as cv

import animeface
import net


def recognize(image):
    model = L.Classifier(net.MyChain())
    serializers.load_npz("animeface.model", model)
    
    x = np.array([image], np.float32)[[0]]
    x = Variable(x)
    y = model.predictor(x)
    ys = list(y.data[0])
    class_id = ys.index(max(ys))
    
    tag2id = animeface.get_class_id_table()
    id2tag = {tag2id[tag]:tag for tag in tag2id}

    print id2tag[class_id]


if __name__ == "__main__":
    if len(os.sys.argv)<2:
        print "Usage: python recognize.py path_to_image"
        sys.exit()
    
    filepath = os.sys.argv[1]
    image = animeface.load_image(filepath)
    recognize(image)
