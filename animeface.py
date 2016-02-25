# coding: utf-8

import os
import random

import numpy as np
import cv2 as cv


DATASET_DIR = "./animeface-character-dataset/thumb"

def unique(l):
    return list(set(l))

def load_dataset():
    dataset = []
    for dir_name in os.listdir(DATASET_DIR):
        tag = dir_name[:3]
        dir_path = os.path.join(DATASET_DIR, dir_name)
        for file_name in os.listdir(dir_path):
            if not file_name.endswith(".png"):
                continue
            file_path = os.path.join(dir_path, file_name)
            image = cv.imread(file_path)
            image = cv.resize(image, (32, 32))
            image = image.transpose(2, 0, 1)
            image = image / 255
            dataset.append((tag, image))

    random.shuffle(dataset)
    
    tags = unique([x[0] for x in dataset])
    tag2id = dict(zip(sorted(tags), range(len(tags))))
    target = np.array([tag2id[x[0]] for x in dataset], np.int32)
    data = np.array([x[1] for x in dataset], np.float32)
    return (data, target)
        

if __name__ == "__main__":
    load_dataset()
