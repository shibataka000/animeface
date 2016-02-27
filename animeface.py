# coding: utf-8

import os
import random

import numpy as np
import cv2 as cv


DATASET_DIR = "./animeface-character-dataset/thumb"
IMAGE_SIZE = 32
N_CLASS = 203


def load_image(path):
    image = cv.imread(path)
    image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.transpose(2, 0, 1)
    image = image / 255
    return image


def load_dataset():
    data = []
    target = []
    tag2id = get_class_id_table()

    for dir_name in os.listdir(DATASET_DIR):
        tag = dir_name
        class_id = tag2id[tag]
        dir_path = os.path.join(DATASET_DIR, dir_name)

        for file_name in os.listdir(dir_path):
            if not file_name.endswith(".png"):
                continue
            file_path = os.path.join(dir_path, file_name)
            image = load_image(file_path)
            data.append(image)
            target.append(class_id)

    data = np.array(data, np.float32)
    target = np.array(target, np.int32)

    return (data, target)


def get_class_id_table():
    tags = os.listdir(DATASET_DIR)
    tags = sorted(tags)
    tag2id = {tags[i]: i for i in range(len(tags))}
    return tag2id
