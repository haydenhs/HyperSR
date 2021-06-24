import argparse
import os
import sys
import random
import time
import torch
import numpy as np
from tqdm import tqdm

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from dataset import HSTrainingData, HSTestData
from losses import SAMLoss


def main():
    # parser
    parser = argparse.ArgumentParser(description="parser for HSI SISR network")
    parser.add_argument("--data_dir", type=str, default='/home/lab611/JiangJJ', help="dataset directory")
    parser.add_argument("--dataset", type=str, default="Chikusei", help="dataset name")
    parser.add_argument("--model", type=str, default="IPNSR", help="model_name")
    parser.add_argument("--sr_factor", type=int, default=4, help="super-resolution factor")
    parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size, default set to 16")
    parser.add_argument("--epochs", type=int, default=11, help="training epochs")
    parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    parser.add_argument("--log_interval", type=int, default=10, help="log interval for printing out info")
    parser.add_argument("--gpus", type=str, default="0, 1", help="gpu ids (default: 7)")
    parser.add_argument("--resume", type=str, help='continue training from a specific checkpoint')

    args = parser.parse_args()
    data_path = os.path.join(args.data_dir, args.dataset + '_x' + str(args.sr_factor))
    args.train_path = os.path.join(data_path, args.dataset + '_train')
    args.eval_path = os.path.join(data_path, args.dataset + '_eval')
    args.model_name = args.dataset + '_' + args.model + '_x' + str(args.sr_factor)
    print("Current training model: {}".format(args.model_name))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    train(args)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('===> Loading datasets')
    train_set = HSTrainingData(image_dir=args.train_path, augment=True)
    eval_set = HSTrainingData(image_dir=args.eval_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True)

    print('===> Building model')
    #net = SSPSR()
    #net = FPNSR(n_channels=128, n_feats=256, n_layers=4)
    net = IPNSR(n_channels=128, n_layers=4, n_iters=5, n_feats=256)
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint from '{}'".format(args.resume))
            start_epoch = int(args.resume[-6:-4]) # only support 10-99 epoch
            net.load_state_dict(args.resume(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    net.to(device).train()

    # Loss functions
    L1_loss = torch.nn.L1Loss()
    #sam_loss = SAMLoss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter('runs/'+ args.model_name + '_' + str(time.ctime()))

    print('===> Start training')
    progress_bar = tqdm(total=(args.epochs - start_epoch) * len(train_set), dynamic_ncols=True)
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e)
        losses = []
        for iteration, (x, lms, gt) in enumerate(train_loader):
            progress_bar.update(n=args.batch_size)
            x = x.to(device)
            lms, gt = lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y = net(x, lms)
            loss = L1_loss(y, gt)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # tensorboard logging
            if (iteration + 1) % args.log_interval == 0:
                progress_bar.set_description("===> {} Epoch[{}]({}/{}): Loss:{:.6f}".format(time.ctime(), e+1, iteration + 1, len(train_loader), loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f} Learning Rate {}".format(time.ctime(), e+1, np.mean(losses), optimizer.param_groups[0]['lr']))
        # run validation every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', np.mean(losses), e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 10 == 0:
            save_checkpoint(args, net, e+1)

    # save model after training
    net.eval().cpu()
    save_model_filename = args.model_name + "_epoch_" + str(args.epochs) + ".pth"
    save_model_path = os.path.join("./trained_models", save_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), save_model_path)
    else:
        torch.save(net.state_dict(), save_model_path)
    progress_bar.close()
    print("\nDone, trained model saved at", save_model_path)


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms = ms.to(device)
            lms, gt = lms.to(device), gt.to(device)
            y = model(ms, lms)
            loss = criterion(y, gt)
            losses.append(loss.item())
        print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), np.mean(losses)))
    # back to training mode
    model.train()
    return np.mean(losses)


def save_checkpoint(args, model, epoch):
    """ Save model checkpoint during training."""
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.model_name + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), ckpt_model_path)
    else:
        torch.save(model.state_dict(), ckpt_model_path)
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
