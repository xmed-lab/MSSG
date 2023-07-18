import os
import random
import numpy as np
from PIL import Image
from multiprocessing import Array, Process


def chunks(lst, num_workers=None, n=None):
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = int(np.ceil(len(lst) / num_workers))
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list


def get_miou_score(pred_image_path, groundtruth_path, num_workers=5, num_class=2):
    image_names = list(map(lambda x: x.split('.')[0], os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            pred_mask = np.asarray(Image.open(pred_image_path + f"/{im_name}.png")).reshape(-1) / 255
            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)
            groundtruth = np.int64(groundtruth > 0).astype(np.uint8)

            gt_list.extend(groundtruth)
            pred_list.extend(pred_mask)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    for i in range(num_class):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
    return total / num_class
