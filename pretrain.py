
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import torch
import argparse
from models.pc_model import Point_MAE
from models.img_model import build_mae_from_cfg
from models.pimae import PiMAE
import torch.nn as nn
from AUdata import BP4DDetectionDatasetNew
from tools.builder import build_opti_sche, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from utilss.AverageMeter import AverageMeter, Acc_Metric
from utilss.util import set_seed, set_log
import pickle
from configs import Config
import numpy as np
import random

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path'
                        )
    args = parser.parse_args()

    return args

def get_new_list(data_path, root_path, new_path):
    with open(root_path, 'r') as infile, open(new_path, 'w') as outfile:
        for line in infile:
            line = line.strip() 
            idx = line.split('/')[-1].split('.')[0]
            random_integer = random.randint(8, 18)
            print(random_integer)
            idx_minus_3 = int(idx) - random_integer
            new_line = line.replace(f"{idx}.pickle", f"{idx_minus_3}.pickle")
            new_filepath = str(data_path + new_line)
            if os.path.exists(new_filepath):
                outfile.write(f"{line} {new_line}\n")
            else:
                print('error')
                print(line, new_line)

def get_new_list_affwild(data_path, root_path, new_path):
    with open(root_path, 'r') as infile, open(new_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            idx = line.split('/')[-1].split('.')[0]
            random_integer = random.randint(8, 18)
            print(random_integer)
            idx_minus_3 = int(idx) - random_integer
            idx_minus_3_str = str(idx_minus_3).zfill(5)
            new_line = line.replace(f"{idx}.pickle", f"{idx_minus_3_str}.pickle")
            new_filepath = os.path.join(data_path, new_line)
            if os.path.exists(new_filepath):
                outfile.write(f"{line} {new_line}\n")
            else:
                print('error')
                print(line, new_line)


def get_dataloader(config):
    train_data = BP4DDetectionDatasetNew(train=True,
                                         list_path=config.new_path,
                                         root_path=config.data_root,
                                         num_points=20000,
                                         use_color=False,
                                         use_height=False,
                                         augment=False,
                                         use_random_cuboid=True,
                                         random_cuboid_min_points=30000,
                                         img_size=config.img_size)
    num_batchs = int(np.ceil(config.SAMPLES / config.max_lens))
    lengths = [config.max_lens] * (num_batchs - 1)
    last_split_size = config.SAMPLES - config.max_lens * (num_batchs - 1)
    lengths.append(last_split_size)
    print(lengths)
    train_split = torch.utils.data.random_split(train_data, lengths)

    train_dataloaders = [(torch.utils.data.DataLoader(x, batch_size=config.batch_size, pin_memory=True,
                                                       num_workers=config.num_workers, shuffle=True)) for x in train_split]
    test_dataloader = None
    return train_dataloaders, test_dataloader


def run_pretrain(new_path):
    # get args
    args = get_args()
    config = Config.fromfile(args.config)

    config.new_path = new_path
    config.model_path = config.file_path + 'model/'
    config.train_results_path = config.file_path + 'results/'
    config.writer_path = config.file_path + 'writer/'

    if not os.path.exists(config.file_path):
        os.makedirs(config.file_path)
    train_writer = SummaryWriter(config.writer_path)

    logger = set_log(config, 'log_new.txt')
    logger.info('Self Supervised Training: {}'.format(config.model_name))
    set_seed(config)
    logger.info('set seed: {}'.format(config.seed))

    # get data
    num_batchs = int(np.ceil(config.SAMPLES / config.max_lens))

    # get model
    pc_branch = Point_MAE(config.pc_model) 
    img_branch = build_mae_from_cfg(config.img_model, norm_pix_loss=config.norm_pix_loss,
                                    img_size=config.img_size) 
    model = PiMAE(pc_branch, img_branch, config.joint_model)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = build_opti_sche(model, config)
    # training
    for epoch in range(0, config.num_epochs):
        # get data
        train_dataloaders, test_dataloader = get_dataloader(config)

        model.train()
        optimizer.zero_grad()

        losses = AverageMeter(['Loss'])
        img_losses = AverageMeter(['Loss'])
        freq_losses = AverageMeter(['Loss'])
        pc_losses = AverageMeter(['Loss'])

        for batch_idx in range(num_batchs):
            print(batch_idx, num_batchs)
            num_iter = 0

            losses_batch = AverageMeter(['Loss'])
            img_losses_batch = AverageMeter(['Loss'])
            freq_losses_batch = AverageMeter(['Loss'])
            pc_losses_batch = AverageMeter(['Loss'])

            for idx, data in enumerate(train_dataloaders[batch_idx]):
                num_iter += 1
                pcs, imgs, pcs_lag, imgs_lag = data
                pcs, imgs, pcs_lag, imgs_lag = pcs.cuda(), imgs.cuda(), pcs_lag.cuda(), imgs_lag.cuda()
                img_loss, freq_loss, pc_loss, save_dict = model(pcs, imgs, pcs_lag, imgs_lag)

                bs = imgs.shape[0]
                img_loss = img_loss.sum() / bs
                freq_loss = freq_loss.sum()
                pc_loss = pc_loss.sum() / bs
                loss = pc_loss + img_loss + freq_loss
                loss.backward()

                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()

                losses.update([loss.item()])
                img_losses.update([img_loss.item()])
                pc_losses.update([pc_loss.item()])
                freq_losses.update([freq_loss.item()])

                losses_batch.update([loss.item()])
                img_losses_batch.update([img_loss.item()])
                pc_losses_batch.update([pc_loss.item()])
                freq_losses_batch.update([freq_loss.item()])

                torch.cuda.empty_cache()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', losses_batch.avg(0), int(epoch * 2 + batch_idx))
                train_writer.add_scalar('Loss/Batch/PC_Loss_1', pc_losses_batch.avg(0), int(epoch * 2 + batch_idx))
                train_writer.add_scalar('Loss/Batch/IMG_Loss_1', img_losses_batch.avg(0), int(epoch * 2 + batch_idx))
                train_writer.add_scalar('Loss/Batch/Freq_Loss_1', freq_losses_batch.avg(0), int(epoch * 2 + batch_idx))

            logger.info(
                '[Training] Batch: %d EPOCH: %d Losses = %s PC_Losses = %s IMG_Losses = %s FREQ_Losses = %s lr = %.6f' %
                (batch_idx, epoch, ['%.4f' % l for l in losses_batch.avg()],
                 ['%.4f' % l for l in pc_losses_batch.val()], ['%.4f' % l for l in img_losses_batch.val()],
                 ['%.4f' % l for l in freq_losses_batch.val()], optimizer.param_groups[0]['lr']))

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/PC_Loss_1', pc_losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/IMG_Loss_1', img_losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Freq_Loss_1', freq_losses.avg(0), epoch)

        logger.info('[Training] EPOCH: %d Losses = %s PC_Losses = %s IMG_Losses = %s FREQ_Losses = %s lr = %.6f' %
                    (epoch, ['%.4f' % l for l in losses.avg()],
                     ['%.4f' % l for l in pc_losses.val()], ['%.4f' % l for l in img_losses.val()],
                     ['%.4f' % l for l in freq_losses.val()], optimizer.param_groups[0]['lr']))
        print('[Training] EPOCH: %d Losses = %s PC_Losses = %s IMG_Losses = %s FREQ_Losses = %s lr = %.6f' %
                    (epoch, ['%.4f' % l for l in losses.avg()],
                     ['%.4f' % l for l in pc_losses.val()], ['%.4f' % l for l in img_losses.val()],
                     ['%.4f' % l for l in freq_losses.val()], optimizer.param_groups[0]['lr']))

        save_checkpoint(model, optimizer, epoch, None, None, str(config.model_path + f'saved-epoch-{epoch:03d}.pkl'),
                        config, logger=None)

    if train_writer is not None:
        train_writer.close()


if __name__ == '__main__':
    data_path = config.data_path
    root_path = config.root_path
    new_path = config.new_path
    get_new_list(data_path, root_path, new_path)  # Step: 1 
    get_new_list_affwild(data_path, root_path, new_path)  # Step: 1 
    run_pretrain(new_path)   # Step: 2: run pretrain

