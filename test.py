import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
from torch.utils.data import DataLoader

from models import *
from metrics import quality_assessment
from dataset import HSTestData


# global settings

test_data_dir = '/home/lab611/JiangJJ/Chikusei_x4/Chikusei_test_x4.mat'
model_name = 'Chikusei_IPNSR_x4'
save_model_title = model_name + '_epoch_11'
save_path = './trained_models/' + save_model_title + '.pth'
result_dir = './result/' + save_model_title + '.mat'


def main():
    # parsers
    parser = argparse.ArgumentParser(description="parser for HyperSR network")
    parser.add_argument("--cuda", type=int, required=False, default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")
    #parser.add_argument("--test_dir", type=str, required=True, help="directory of testset")
    #parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    test(args)


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    logger = pd.DataFrame()
    with torch.no_grad():
        # loading model
        model = IPNSR(n_channels=128, n_layers=4, n_iters=5, n_feats=256)
        #model = SSPSR()
        model.load_state_dict(torch.load(save_path))
        model.to(device).eval()
        result = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = model(ms, lms)
            #y = y.squeeze().permute(1,2,0).cpu().numpy()
            #gt = gt.squeeze().permute(1,2,0).cpu().numpy()
            y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
            gt = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            indices = quality_assessment(gt, y, data_range=1., ratio=8)
            logger = logger.append([indices], ignore_index=True)
            result.append(y)
        np.stack(result)
        sio.savemat(model_name + '.mat', {'output':result})
        logger.to_csv(model_name + '.csv')
    print("===> {}\t Testing RGB Complete: Avg. RMSE: {:.4f}, Avg. MPSNR: {:.2f}, Avg. MSSIM: {:.2f}".format(
            time.ctime(), logger['RMSE'].mean(), logger['MPSNR'].mean(), logger['MSSIM'].mean()))


if __name__ == "__main__":
    main()
