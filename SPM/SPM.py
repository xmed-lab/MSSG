import argparse

import skimage
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import numpy as np
import torch.nn.init
import glob
from torch.autograd import Variable
from sklearn.cluster import KMeans
from tqdm import tqdm
from loss import SpatialLoss
from model import CSNet
from metric import get_miou_score

parser = argparse.ArgumentParser(
    description='SPM Module for Border Proposal Mining')

parser.add_argument('--gpu', default=1, type=int, help='whether to use gpu')
parser.add_argument('--gpu_ids', default=[0], nargs='+', type=int)

parser.add_argument('--train_image_path', default='../glas/training_images', type=str, help='training img path')
parser.add_argument('--train_ground_truth_path', default='../glas/training_gts', type=str,
                    help='training ground truth path')

parser.add_argument('--output_root', help='output root folder path',
                    default='result')

parser.add_argument('--nConv', metavar='M', default=3, type=int, help='number of convolutional layers')
parser.add_argument('--nChannel', metavar='N', default=128, type=int, help='number of channels')
parser.add_argument('--cluster_num', default=5, type=int, help='number of clusters')

parser.add_argument('--maxIter', metavar='T', default=50, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.01, type=float,
                    help='learning rate')

parser.add_argument('--stepsize_ce', metavar='CE', default=1, type=float,
                    help='step size for cross entropy loss', required=False)
parser.add_argument('--stepsize_ss', metavar='SS', default=5, type=float,
                    help='step size for sparse spatial loss')

args = parser.parse_args()


def load_image(image_path, use_cuda=True):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    image = torch.from_numpy(np.array([image.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        image = image.cuda()
    return h, w, Variable(image)


def init_model(channel=3, use_cuda=True):
    model = CSNet(args, channel)
    if use_cuda:
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu_ids, output_device=args.gpu_ids[0])
        model = model.cuda()
    model.train()
    return model


def train_with_one_image(max_iter, optimizer, model, image, h, w, loss_ce, loss_spatial):
    for _ in tqdm(range(max_iter)):
        optimizer.zero_grad()
        output = model(args, image)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

        outputmp = output.reshape((h, w, args.nChannel))

        _, target = torch.max(output, 1)

        loss = args.stepsize_ce * loss_ce(output, target) + args.stepsize_ss * loss_spatial(outputmp)
        loss.backward()
        optimizer.step()
        # print(batch_idx, '/', args.maxIter, '|', ' | loss :', loss.item())
        # if loss.item() <= 0.5:
        #     break


def get_highest_gray(original_img, mask, threshold=10000):
    average_gray_list = []
    cluster_list = np.unique(mask)
    single_level_mask = mask

    for c in cluster_list:
        binary_mask = np.where(single_level_mask == c, 1, 0)
        img_gray_matrix = original_img * binary_mask
        average_gray_list.append(np.sum(img_gray_matrix) / np.sum(binary_mask))
    arranged_index = np.argsort(average_gray_list)
    for min_index in arranged_index:
        max_gray_cluster_index = min_index
        max_gray = np.where(single_level_mask == cluster_list[max_gray_cluster_index], 1, 0)
        if np.sum(np.int64(max_gray > 0)) > threshold:
            break
    return max_gray


def erosion_process(img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    erosion = cv2.erode(img, kernel, iterations=iterations)
    return erosion


def deflate_process(img, kernel_size=5, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilate = cv2.dilate(img, kernel, iterations)
    return dilate


def remove_small_area(mask):
    binary_mask = np.int64(mask > 0)
    h, w = binary_mask.shape
    if np.sum(binary_mask) > (0.7 * h * w):
        erode_mask = erosion_process(mask)
        mask = skimage.morphology.remove_small_objects((erode_mask > 0), min_size=1000)
    elif np.sum(binary_mask) > (0.45 * h * w):
        erode_mask = erosion_process(mask)
        mask = skimage.morphology.remove_small_objects((erode_mask > 0), min_size=600)
    elif np.sum(binary_mask) > (0.25 * h * w):
        mask = skimage.morphology.remove_small_objects((mask > 0), min_size=400)
    elif np.sum(binary_mask) > (0.2 * h * w):
        mask = skimage.morphology.remove_small_objects((mask > 0), min_size=100)
    elif np.sum(binary_mask) > (0.1 * h * w):
        mask = skimage.morphology.remove_small_objects((mask > 0), min_size=50)
    return mask


def fill_middle_holes(mask):
    binary_mask = np.int64(mask > 0)
    h, w = binary_mask.shape
    if np.sum(binary_mask) < (0.25 * h * w):
        mask = deflate_process(mask)
    removed_noise_mask = skimage.morphology.remove_small_holes(mask, area_threshold=15000)
    return removed_noise_mask


def main():
    image_list = sorted(glob.glob(args.train_image_path + '/*'))

    cluster_label_path = os.path.join(args.output_root, f'cluster_{args.cluster_num}_nConv_{args.nConv}/cluster_label')
    border_proposal_path = os.path.join(args.output_root,
                                        f'cluster_{args.cluster_num}_nConv_{args.nConv}/border_proposal')
    gland_proposal_path = os.path.join(args.output_root,
                                       f'cluster_{args.cluster_num}_nConv_{args.nConv}/gland_proposal')

    if not os.path.exists(border_proposal_path):
        os.makedirs(border_proposal_path)
    if not os.path.exists(gland_proposal_path):
        os.makedirs(gland_proposal_path)
    if not os.path.exists(cluster_label_path):
        os.makedirs(cluster_label_path)

    # for each image
    for image_path in image_list:
        h, w, data = load_image(image_path)

        # Load the model
        model = init_model(channel=data.size(1), use_cuda=args.gpu)

        # Cross-Entropy loss definition
        loss_ce = torch.nn.CrossEntropyLoss().cuda()
        loss_spatial = SpatialLoss(h, w, args.nChannel).cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # Training
        train_with_one_image(max_iter=args.maxIter, optimizer=optimizer, model=model,
                             image=data, h=h, w=w, loss_ce=loss_ce, loss_spatial=loss_spatial)

        # inference and post-process
        output = model(args, data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
        cluster_label = KMeans(n_clusters=args.cluster_num, random_state=9).fit_predict(output.detach().cpu().numpy())

        label_colours = np.random.randint(255, size=(args.cluster_num, 3))
        cluster_label_rgb = np.array([label_colours[c % args.cluster_num] for c in np.array(cluster_label)]).reshape(h, w, 3).astype(np.uint8)

        file_name = image_path.split('/')[-1]

        cv2.imwrite(cluster_label_path + f'/{file_name}', np.int8(cluster_label_rgb))

        origin_image = cv2.imread(image_path, 0)
        border_proposal = get_highest_gray(original_img=origin_image, mask=np.array(cluster_label.reshape(h, w).astype(np.uint8)))

        cv2.imwrite(border_proposal_path + f'/{file_name}', np.int8(border_proposal))

        # border_proposal = cv2.imread(border_proposal_path + f'/{file_name}', 0)

        removed_noise_tissue_mask = remove_small_area(border_proposal)
        filled_middle_holes_mask = fill_middle_holes(removed_noise_tissue_mask.astype(np.uint8))
        filled_middle_holes_mask = np.int8(filled_middle_holes_mask > 0)

        gland_proposal = border_proposal + filled_middle_holes_mask

        cv2.imwrite(gland_proposal_path + f'/{file_name}', np.int64(gland_proposal))


if __name__ == '__main__':
    main()
