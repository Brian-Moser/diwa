import os
import json
import sys
import glob
import cv2
import numpy as np
from os import path as osp
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from pathlib import Path


def extract_subimages(input_folder, save_folder, crop_size, step, thresh_size, n_thread=20):
    input_folder = input_folder
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    # scan all images
    img_list = list(glob.glob(input_folder + '**/*', recursive=True))
    img_list = [f for f in img_list if os.path.isfile(f)]

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker, args=(path, crop_size, step, thresh_size, save_folder),
                         callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, crop_size, step, thresh_size, save_folder, compression_level=3):
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(save_folder, f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    process_info = f'Processing {img_name} ...'
    return process_info


def get_working_directory():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def create_data_list(folders, output_folder, ds_name, crop_size=96, step=86, thresh_size=0):
    print("\nCreating data list <" + ds_name + ">...")
    images = list()
    sub_target_folder = output_folder + ds_name + "/"
    for d in folders:
        extract_subimages(d, sub_target_folder, crop_size, step, thresh_size)
    folders = [sub_target_folder]
    for d in folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            images.append(img_path)
    print("Total amount: %d images" % len(images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='./dataset/DIV2K_train_HR'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/')
    args = parser.parse_args()
    create_data_list(folders=[args.path],
                     output_folder=args.out,
                     ds_name='div2kLL',
                     crop_size=192,
                     step=48,
                     thresh_size=0)

