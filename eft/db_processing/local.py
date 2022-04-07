import os
from os.path import join
import sys
import json
import numpy as np

# For debugging
from renderer import viewer2D  # , glViewer
import cv2
# from .read_openpose import read_openpose
# from read_openpose import read_openpose
from eft.db_processing.read_openpose import read_openpose


def coco_extract(dataset_path, openpose_path, out_path, bWithCOCOFoot=False):
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_, annot_ids_ = [], [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 'annotations', 'dance_0406.json')

    print("Processing COCO Dataset")
    # print(f"image dir: {cocoImgDir}")
    print(f"annot dir: {json_path}")

    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # print(annot)
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17, 3))
        keypoints[keypoints[:, 2] > 0, 2] = 1
        # check if all major body joints are annotated

        # Change the following to select a subset of coco
        if sum(keypoints[5:, 2] > 0) < 12:  # Original: cases that all body limbs are annotated
            continue

        # image name
        image_id = annot['image_id']
        annot_id = annot['id']

        # img_name = str(imgs[image_id]['file_name'])
        # print(img_name)
        # raise '111'
        #
        # img_name_full = join('datasets', img_name)
        img_name_full = str(imgs[image_id]['path'])[1:].replace('bobing_0331_out_0406/', '')

        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = keypoints
        # scale and center
        bbox = annot['bbox']  # X,Y,W,H
        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        scale = scaleFactor * max(bbox[2], bbox[3]) / 200

        # read openpose detections
        # json_file = os.path.join(openpose_path, 'coco',
        #     img_name.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'coco')
        openpose = np.zeros([25, 3])  # blank

        # store data
        imgnames_.append(img_name_full)
        annot_ids_.append(annot_id)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)


    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'dance_0406.npz')

    print(f"Saving pre-processed db output: {out_file}")

    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             openpose=openposes_,
             annotIds=annot_ids_)


if __name__ == '__main__':
    coco_extract(dataset_path='./data_sets/dance_0406/', openpose_path=None, out_path='./preprocessed_db/')
