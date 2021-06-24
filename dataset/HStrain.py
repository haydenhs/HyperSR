import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import utils
import cv2


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, augment=None):
        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = []
        for i in self.image_folders:
            images = os.listdir(i)
            for j in images:
                if is_mat_file(j):
                    full_path = os.path.join(i, j)
                    self.image_files.append(full_path)
        self.augment = augment
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        ms = np.array(data['ms'][...], dtype=np.float32)
        lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)
        '''
        # add Laplacian pyramid
        gp = []
        for _ in range(3):
            gp.append(ms.copy())
            ms = cv2.pyrDown(ms)
        ss = []
        for lvl in gp:
            ss.append(torch.from_numpy(lvl).permute(2, 0, 1))
        '''
        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor
