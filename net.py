# coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import animeface


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(animeface.IMAGE_SIZE * animeface.IMAGE_SIZE * 3, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, animeface.N_CLASS),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
