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
            l1=L.Linear(32*32*3, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 176),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


def train():
    model = L.Classifier(MyChain())
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    (data, target) = animeface.load_dataset()
    N = int(len(data) * 0.8)
    N_test = len(data) - N
    batchsize = 100
    
    x_train, x_test = np.split(data, [N])
    y_train, y_test = np.split(target, [N])
    
    for epoch in range(20):
        print "epoch {0}".format(epoch)
        
        indexes = np.random.permutation(N)
        sum_loss, sum_accuracy = 0, 0
        for i in range(0, N, batchsize):
            x = Variable(x_train[indexes[i : i + batchsize]])
            t = Variable(y_train[indexes[i : i + batchsize]])
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * batchsize
            sum_accuracy += float(model.accuracy.data) * batchsize
        print "train loss={0}, accuracy={1}".format(sum_loss/N, sum_accuracy/N)

        sum_loss, sum_accuracy = 0, 0
        for i in range(0, N_test, batchsize):
            x = Variable(x_test[i : i + batchsize])
            t = Variable(y_test[i : i + batchsize])
            loss = model(x, t)
            sum_loss += float(loss.data) * batchsize
            sum_accuracy += float(model.accuracy.data) * batchsize
        print "test loss={0}, accuracy={1}".format(sum_loss/N_test, sum_accuracy/N_test)

    serializers.save_npz("animeface.model", model)
        

def main():
    train()

if __name__ == "__main__":
    main()
