# coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import animeface
import net

def recognize():
    model = L.Classifier(net.MyChain())
    # optimizer = optimizers.SGD()
    # optimizer.setup(model)

    (data, target) = animeface.load_dataset()

    serializers.load_npz("animeface.model", model)
    # serializers.load_npz("animeface.state", optimizer)

    total = {}
    ok = {}
    for i in range(176):
        total[i] = 0
        ok[i] = 0

    for i in range(len(data)):
        x = Variable(data[[i]])
        y = model.predictor(x)
        yy = list(y.data[0])
        actual = yy.index(max(yy))
        expected = target[i]
        print "{0} is expected, but actual {1}".format(expected, actual)
        total[expected] += 1
        if expected == actual:
            ok[expected] += 1

    id2tag = animeface.get_tag_dir()

    for i in range(176):
        print id2tag[i], float(ok[i]) / float(total[i])

if __name__ == "__main__":
    recognize()
