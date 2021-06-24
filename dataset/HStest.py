import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch


class HSTestData(data.Dataset):
    def __init__(self, image_dir):
        test_data = sio.loadmat(image_dir)
        self.ms = np.array(test_data['ms'][...], dtype=np.float32)
        self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
        self.gt = np.array(test_data['gt'][...], dtype=np.float32)

    def __getitem__(self, index):
        gt = self.gt[index, :, :, :]
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]
        ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        return ms, lms, gt

    def __len__(self):
        return self.gt.shape[0]
