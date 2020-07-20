import argparse
from argparse import Namespace
import os
from datetime import datetime
from math import log10
import mlflow

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
from ..pytorch_utils import normalize

import pytorch_ssim
from src.srgan.data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from src.srgan.srgan_loss import GeneratorLoss
from src.srgan.srgan_module import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')
parser.add_argument('--batch_size', default=1, type=int, help='batch size for training')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--epochs', default=60, type=int, help='train epoch number')
parser.add_argument('--train_set', default='data/interim/datasets/train', type=str,
                    help='path to the training data set')
parser.add_argument('--test_set', default='data/interim/datasets/test', type=str, help='path to the test data set')
parser.add_argument('--val_set', default='data/interim/datasets/val', type=str, help='path to the val data set')


parser.add_argument('-i', '--gen_images', nargs='?', const=True, default=False, type=bool,
                    help='If enabled store images on disk for each epoch.')
parser.add_argument('--eval',  help='Path to Generator model state dict.')


opt: Namespace = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
if opt.eval:
    opt.epochs = 0
NUM_EPOCHS = opt.epochs
BATCH_SIZE = opt.batch_size

train_set = TrainDatasetFromFolder(opt.train_set, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
test_set = ValDatasetFromFolder(opt.test_set, upscale_factor=UPSCALE_FACTOR)
val_set =ValDatasetFromFolder(opt.val_set, upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1 if not opt.eval else BATCH_SIZE, shuffle=False)



netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

if opt.eval:
    netG.load_state_dict(torch.load(opt.eval))


generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


def log_results_mlflow(epoch, **kwargs):
    for k, v in kwargs.items():
        mlflow.log_metric(k, v, epoch)


def log_config_mlflow(config: Namespace):
    for k, v in vars(config).items():
        mlflow.log_param(k, v)


mlflow.set_experiment('sr_eval' if opt.eval else 'srgan')


def eval(data_loader):
    netG.eval()
    progress_bar = tqdm(data_loader)
    ssims, psnrs = [],[]
    images = []
    for lr, hr, val_hr in progress_bar:
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = val_hr.cuda()

        sr = netG(lr)


        for pred, img in zip(sr,hr):
            pred = normalize(pred)
            mse = ((pred -img)**2).data.mean()
            psnrs.append(10*log10(1**2/mse))
            ssims.append(0)#ssims.append(pytorch_ssim.ssim(pred, img).data)

        progress_bar.set_description(
            desc='[Evaluation] PSNR: %.4f dB SSIM: %.4f' % (
                mean(psnrs), mean(ssims)))

        if opt.gen_images:
            images.extend(
            [display_transform()(hr.cpu().squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])

    return mean(ssims), mean(psnrs), images



with mlflow.start_run() as run:
    print(opt)
    log_config_mlflow(opt)
    print("Started mlflow run '{!s}' in experiment '{!s}'.".format(run.info.run_id, 'sr_eval' if opt.eval else 'srgan'))

    epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)

        d_losses, g_losses, d_scores, g_scores = [], [],[], []
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = len(data)
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            d_loss = 1 - real_out + fake_out

            g_losses.extend([g_loss.detach().item()]*batch_size)
            d_losses.extend([d_loss.detach().item()]*batch_size)
            d_scores.extend([real_out.detach().item()] * batch_size)
            g_scores.extend([fake_out.detach().item()] * batch_size)

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS,
                mean(d_losses), mean(g_losses), mean(d_scores), mean(g_scores)
            ))
        log_results_mlflow(
            epoch,
            train_d_loss=mean(d_losses), train_d_score=mean(d_scores),
            train_g_loss=mean(g_losses), train_g_score=mean(g_scores)
        )

        # save loss\scores\psnr\ssim
        results['d_loss'].append(mean(d_losses))
        results['g_loss'].append(mean(g_losses))
        results['d_score'].append(mean(d_scores))
        results['g_score'].append(mean(g_scores))

        out_path = 'models/srgan/{}'.format(run.info.run_id)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # save model parameters
        epoch_path_gen = '{}/checkpoints'.format(out_path)
        epoch_path_dis = '{}/checkpoints_dis'.format(out_path)
        if not os.path.exists(epoch_path_gen):
            os.makedirs(epoch_path_gen)
        if not os.path.exists(epoch_path_dis):
            os.makedirs(epoch_path_dis)
        torch.save(netG.state_dict(), '%s/model_epoch_%d.pth' % (epoch_path_gen, epoch))
        torch.save(netD.state_dict(), '%s/model_epoch_%d.pth' % (epoch_path_dis, epoch))


        val_ssim, val_psnr, val_images = eval(val_loader)
        results['psnr'].append(val_psnr)
        results['ssim'].append(val_ssim)
        log_results_mlflow(epoch, val_avg_psnr=val_psnr, val_avg_ssim=val_ssim)

        # save validation results
        if opt.gen_images:
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1

            if not os.path.exists(out_path + '/results'):
                os.makedirs(out_path + '/results')

            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, '%s/results/epoch_%d_index_%d.png' % (out_path, epoch, index), padding=5)
                index += 1


        if epoch % 10 == 0 and epoch != 0:
            stat_path = '{}/statistics/'.format(out_path)
            if not os.path.exists(stat_path):
                os.makedirs(stat_path)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(stat_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

    test_ssim, test_psnr, test_images = eval(test_loader)
    log_results_mlflow(epoch, test_avg_psnr=test_psnr, test_avg_ssim=test_ssim)
