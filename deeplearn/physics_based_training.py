# -*- coding: utf-8 -*-
""" Trains a physics-based model
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import os
import numpy as np
import time
import datetime
import sys
sys.path.append('../')
import csv
import torch

from fileio.yamlio import load_config

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from deeplearn.models import GeneratorUNet256_4x, GeneratorUNet64US, GeneratorUNet128_x2
from deeplearn.models import GeneratorResNet
from deeplearn.models import Discriminator, DiscriminatorSN, weights_init_normal
from deeplearn.datasets import ImageDatasetLSM, ImageDatasetRef
from deeplearn.perceptual_loss import VGGPerceptualLoss


# =============================================================================
# USER CONFIG
# =============================================================================
class Config():
    def __init__(self):
        # Config class, adressable as config.property
        self.epoch = 0
        self.n_epochs = 301
        self.batch_size = 8
        self.lrG = 0.0001
        self.lrD = 0.0004
        self.b1 = 0.5
        self.b2 = 0.999
        self.lambda_pixel = 30
        self.beta_pixel = 0.0001
        self.gamma_pixel = 1
        self.decay_epoch = 100
        self.n_cpu = 0
        self.img_size = 64
        self.channels = 1
        self.sample_interval = 500
        self.checkpoint_interval = 100
        self.spectral_norm = False
        self.dual_D = True
        self.smooth_labels = True
        self.model = 'resnet'
        self.scheduler = False
        self.loss_decay = False

    def overwrite_defaults(self, config):
        self.n_epochs = config['n_epochs'] if 'n_epochs' in config else self.n_epochs
        self.b1 = config['b1'] if 'b1' in config else self.b1
        self.b2 = config['b2'] if 'b2' in config else self.b2
        self.model = config['model'] if 'model' in config else self.model
        self.lambda_pixel = config['lambda_pixel'] if 'lambda_pixel' in config else self.lambda_pixel
        self.beta_pixel = config['beta_pixel'] if 'beta_pixel' in config else self.beta_pixel
        self.gamma_pixel = config['gamma_pixel'] if 'gamma_pixel' in config else self.gamma_pixel
        self.smooth_labels = config['smooth_labels'] if 'smooth_labels' in config else self.smooth_labels
        self.spectral_norm = config['spectral_norm'] if 'spectral_norm' in config else self.spectral_norm
        self.dual_D = config['dual_D'] if 'dual_D' in config else self.dual_D
        self.lrD = config['lrD'] if 'lrD' in config else self.lrD
        self.lrG = config['lrG'] if 'lrG' in config else self.lrG
        self.scheduler = config['scheduler'] if 'scheduler' in config else self.scheduler
        self.loss_decay = config['loss_decay'] if 'loss_decay' in config else self.loss_decay


# =============================================================================
# TRAIN NETWORK
# =============================================================================
def train_model(conf, datadir):
    config, _, _ = load_config(datadir + '/TrainConfig.yml')
    dataroot = datadir

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Prepare folders
    imagedir = os.path.join(dataroot, 'training_images')
    imagedir_real = os.path.join(dataroot, 'training_images_real')
    modeldir = os.path.join(dataroot, 'saved_models')

    os.makedirs(imagedir, exist_ok=True)
    os.makedirs(imagedir_real, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_saliency = VGGPerceptualLoss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = conf.lambda_pixel
    beta_pixel = conf.beta_pixel
    gamma_pixel = conf.gamma_pixel

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, conf.img_size // 2 ** 4, conf.img_size // 2 ** 4)

    # Initialize generator and discriminator
    if conf.img_size == 64:
        if conf.model.lower() == 'unet':
            generator = GeneratorUNet64US(in_channels=conf.channels,
                                          out_channels=conf.channels)
        elif conf.model.lower() == 'resnet':
            generator = GeneratorResNet(in_channels=conf.channels,
                                        out_channels=conf.channels)
        else:
            print("No compatible model specified")
            return
    elif conf.img_size == 128:
        generator = GeneratorUNet128_x2(in_channels=conf.channels,
                                        out_channels=conf.channels)
    elif conf.img_size == 256:
        generator = GeneratorUNet256_4x(in_channels=conf.channels,
                                        out_channels=conf.channels)
    else:
        print("No model for image size specified")
        return

    if conf.spectral_norm:
        discriminator = DiscriminatorSN(in_channels=conf.channels)
    else:
        discriminator = Discriminator(in_channels=conf.channels)

    if conf.dual_D:
        if conf.spectral_norm:
            discriminator2 = DiscriminatorSN(in_channels=conf.channels)
        else:
            discriminator2 = Discriminator(in_channels=conf.channels)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        criterion_saliency.cuda()
        if conf.dual_D:
            discriminator2.cuda()

    if conf.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(
            os.path.join(modeldir, 'generator_%d.pth' % conf.epoch)))
        discriminator.load_state_dict(torch.load(
            os.path.join(modeldir, 'discriminator_%d.pth' % conf.epoch)))
        if conf.dual_D:
            discriminator2.load_state_dict(torch.load(
                os.path.join(modeldir, 'discriminator2_%d.pth' % conf.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        if conf.dual_D:
            discriminator2.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=conf.lrG,
                                   betas=(conf.b1, conf.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=conf.lrD,
                                   betas=(conf.b1, conf.b2))
    if conf.dual_D:
        optimizer_D2 = torch.optim.Adam(discriminator2.parameters(),
                                        lr=conf.lrD,
                                        betas=(conf.b1, conf.b2))

    # Rate schedule
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_G, milestones=[150, 250], gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_D, milestones=[150, 250], gamma=0.1)
    if conf.dual_D:
        scheduler_D2 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_D2, milestones=[150, 250], gamma=0.1)

    # =============================================================================
    # Configure dataloaders
    # =============================================================================
    transforms_ = [
        transforms.CenterCrop((conf.img_size, conf.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    # Simulation train
    dataset_train = ImageDatasetLSM(
        root=dataroot,
        transforms_=transforms_,
        mode='train'
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=len(dataset_train),
        shuffle=True,
        num_workers=conf.n_cpu,
    )
    # Simulation validation
    dataset_val = ImageDatasetLSM(
        root=dataroot,
        transforms_=transforms_,
        mode='val'
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=len(dataset_val),
        shuffle=False,
        num_workers=0,
    )
    dataloader_val_display = DataLoader(
        dataset=dataset_val,
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )

    # Real train
    dataset_real = ImageDatasetRef(
        root=dataroot,
        transforms_=transforms_,
        mode='real'
    )
    dataloader_real = DataLoader(
        dataset=dataset_real,
        batch_size=len(dataset_real),
        shuffle=True,
        num_workers=0,
    )
    # Real validation
    dataset_val_real = ImageDatasetRef(
        root=dataroot,
        transforms_=transforms_,
        mode='real'
    )
    dataloader_val_real = DataLoader(
        dataset=dataset_val_real,
        batch_size=len(dataset_val_real),
        shuffle=False,
        num_workers=0,
    )
    dataloader_val_real_display = DataLoader(
        dataset=dataset_val_real,
        batch_size=12,
        shuffle=False,
        num_workers=0,
    )

    # =============================================================================
    # Visualisation
    # =============================================================================
    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(dataloader_val_display))
        real_LR = Variable(imgs["LR"].type(Tensor))
        real_HR = Variable(imgs["HR"].type(Tensor))
        fake_HR = generator(real_LR)
        img_sample = torch.cat((real_HR.data, fake_HR.data, real_LR.data), -2)
        save_image(img_sample,
                   imagedir + "/%s.png" % (batches_done),
                   nrow=6,
                   normalize=True)

    def sample_images_real(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(dataloader_val_real_display))
        real_LR = Variable(imgs["REF"].type(Tensor))
        fake_HR = generator(real_LR)
        img_sample = torch.cat((real_LR.data, fake_HR.data), -2)
        save_image(img_sample,
                   imagedir_real + "/%s.png" % (batches_done),
                   nrow=6,
                   normalize=True)

    class Reporter:
        def __init__(self):
            self.G_loss_train = []
            self.D_loss_train = []
            self.pix_loss_train = []
            self.adv_loss_train = []
            self.sal_loss_train = []
            self.G_loss_val = []
            self.D_loss_val = []
            self.pix_loss_val = []
            self.adv_loss_val = []
            self.sal_loss_val = []
            self.G_loss = []
            self.D_loss = []
            self.pix_loss = []
            self.adv_loss = []
            self.sal_loss = []

        def record(self, dloss, gloss, pix, adv, sal):
            self.D_loss.append(dloss)
            self.G_loss.append(gloss)
            self.pix_loss.append(pix)
            self.adv_loss.append(adv)
            self.sal_loss.append(sal)

        def mean(self, data):
            return sum(data) / len(data)

        def push(self, where):
            if where == 'train':
                self.D_loss_train.append(self.mean(self.D_loss))
                self.G_loss_train.append(self.mean(self.G_loss))
                self.pix_loss_train.append(self.mean(self.pix_loss))
                self.adv_loss_train.append(self.mean(self.adv_loss))
                self.sal_loss_train.append(self.mean(self.sal_loss))
            elif where == 'val':
                self.D_loss_val.append(self.mean(self.D_loss))
                self.G_loss_val.append(self.mean(self.G_loss))
                self.pix_loss_val.append(self.mean(self.pix_loss))
                self.adv_loss_val.append(self.mean(self.adv_loss))
                self.sal_loss_val.append(self.mean(self.sal_loss))

            self.G_loss = []
            self.D_loss = []
            self.pix_loss = []
            self.adv_loss = []
            self.sal_loss = []

        def normalize(self):
            self.D_lossN = self.D_loss / np.max(self.D_loss)
            self.G_lossN = self.G_loss / np.max(self.G_loss)
            self.pix_lossN = self.pix_loss / np.max(self.pix_loss)
            self.adv_lossN = self.adv_loss / np.max(self.adv_loss)
            self.sal_lossN = self.sal_loss / np.max(self.sal_loss)

    # =========================================================================
    # Training
    # =========================================================================
    prev_time = time.time()
    time_left = 0

    report = Reporter()
    fig, ax = plt.subplots()

    full_batch = next(iter(dataloader_train))
    real_batch = next(iter(dataloader_real))
    val_batch = next(iter(dataloader_val))
    val_real_batch = next(iter(dataloader_val_real))

    n_batches = (len(dataset_train) // conf.batch_size)
    n_batches_real = (len(dataset_real) // conf.batch_size)
    n_batches_val = (len(dataset_val) // conf.batch_size)
    n_batches_val_real = (len(dataset_val_real) // conf.batch_size)

    # =========================================================================
    # Training
    # =========================================================================
    def train(i, time_left, epoch):
        # Data load
        batch_LR = full_batch["LR"][i * conf.batch_size:
                                    min((i + 1) * conf.batch_size, len(dataset_train))]
        batch_HR = full_batch["HR"][i * conf.batch_size:
                                    min((i + 1) * conf.batch_size, len(dataset_train))]
        j = i % n_batches_real
        batch_real = real_batch["REF"][j * conf.batch_size:
                                       min((j + 1) * conf.batch_size, len(dataset_real))]

        # Model inputs
        real_sim_LR = Variable(batch_LR.type(Tensor))
        real_sim_HR = Variable(batch_HR.type(Tensor))
        real_real = Variable(batch_real.type(Tensor))

        # Adversarial ground truths
        if conf.smooth_labels:
            valid = Variable(Tensor(0.9 * np.ones((real_sim_LR.size(0), *patch))),
                             requires_grad=False)
            fake = Variable(Tensor(0.0 * np.ones((real_sim_LR.size(0), *patch))),
                            requires_grad=False)
        else:
            valid = Variable(Tensor(np.ones((real_sim_LR.size(0), *patch))),
                             requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_sim_LR.size(0), *patch))),
                            requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()

        # Train simulated data
        fake_sim_HR = generator(real_sim_LR)
        #  Train saliency
        fake_real = generator(real_real)

        # Adversarial GAN loss
        pred_fake = discriminator(fake_sim_HR, real_sim_LR)
        loss_GAN = criterion_GAN(pred_fake, valid)
        if conf.dual_D:
            pred_fake2 = discriminator2(fake_sim_HR, real_sim_LR)
            loss_GAN2 = criterion_GAN(pred_fake2, valid)
            # Criteria
            # loss_GAN = torch.min(loss_GAN, loss_GAN2)
            loss_GAN = torch.max(loss_GAN, loss_GAN2)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_sim_HR, real_sim_HR)

        # Saliency loss
        loss_saliency = criterion_saliency(fake_real, real_real)

        # Backpropagate losses
        if conf.loss_decay:
            loss_decay = 2 * (1 - epoch / conf.n_epochs)
        else:
            loss_decay = 1
        loss_G = loss_decay * (gamma_pixel * loss_GAN +
                               lambda_pixel * loss_pixel +
                               beta_pixel * loss_saliency)
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_sim_HR, real_sim_LR)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_sim_HR.detach(), real_sim_LR)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = loss_decay * (0.5 * (loss_real + loss_fake))
        loss_D.backward()
        optimizer_D.step()

        if conf.dual_D:
            optimizer_D2.zero_grad()
            # Real loss
            pred_real = discriminator2(real_sim_HR, real_sim_LR)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss
            pred_fake = discriminator2(fake_sim_HR.detach(), real_sim_LR)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D2 = loss_decay * (0.5 * (loss_real + loss_fake))
            loss_D2.backward()
            optimizer_D2.step()
            loss_D = 0.5 * (loss_D + loss_D2)

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] [S loss: %f] ETA: %s"
            % (
                epoch + 1,
                conf.n_epochs,
                i + 1,
                n_batches,
                loss_D.item() / loss_decay,
                loss_G.item() / loss_decay,
                loss_pixel.item() * lambda_pixel,
                loss_GAN.item() * gamma_pixel,
                loss_saliency.item() * beta_pixel,
                time_left,
            )
        )

        # Record values
        report.record(loss_D.item() / loss_decay,
                      loss_G.item() / loss_decay,
                      loss_pixel.item() * lambda_pixel,
                      loss_GAN.item() * gamma_pixel,
                      loss_saliency.item() * beta_pixel)

    def validate(i, time_left, epoch):
        # Data load
        batch_LR = val_batch["LR"][i * conf.batch_size:
                                   min((i + 1) * conf.batch_size, len(dataset_val))]
        batch_HR = val_batch["HR"][i * conf.batch_size:
                                   min((i + 1) * conf.batch_size, len(dataset_val))]
        j = i % n_batches_val_real
        batch_real = val_real_batch["REF"][j * conf.batch_size:
                                           min((j + 1) * conf.batch_size, len(dataset_val_real))]

        # Model inputs
        real_sim_LR = Variable(batch_LR.type(Tensor))
        real_sim_HR = Variable(batch_HR.type(Tensor))
        real_real = Variable(batch_real.type(Tensor))

        # Adversarial ground truths
        if conf.smooth_labels:
            valid = Variable(Tensor(0.9 * np.ones((real_sim_LR.size(0), *patch))),
                             requires_grad=False)
            fake = Variable(Tensor(0.0 * np.ones((real_sim_LR.size(0), *patch))),
                            requires_grad=False)
        else:
            valid = Variable(Tensor(np.ones((real_sim_LR.size(0), *patch))),
                             requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_sim_LR.size(0), *patch))),
                            requires_grad=False)

        # ------------------
        #  Validate Generator
        # ------------------
        # Train simulated data
        fake_sim_HR = generator(real_sim_LR)
        #  Train saliency
        fake_real = generator(real_real)

        # Adversarial GAN loss
        pred_fake = discriminator(fake_sim_HR, real_sim_LR)
        loss_GAN = criterion_GAN(pred_fake, valid)
        if conf.dual_D:
            pred_fake2 = discriminator2(fake_sim_HR, real_sim_LR)
            loss_GAN2 = criterion_GAN(pred_fake2, valid)
            # Criteria
            loss_GAN = torch.min(loss_GAN, loss_GAN2)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_sim_HR, real_sim_HR)

        # Saliency loss
        loss_saliency = criterion_saliency(fake_real, real_real)

        # Backpropagate losses
        loss_G = gamma_pixel * loss_GAN + lambda_pixel * loss_pixel + beta_pixel * loss_saliency

        # ---------------------
        #  Validate Discriminator
        # ---------------------
        # Real loss
        pred_real = discriminator(real_sim_HR, real_sim_LR)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_sim_HR.detach(), real_sim_LR)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        if conf.dual_D:
            # Real loss
            pred_real = discriminator2(real_sim_HR, real_sim_LR)
            loss_real = criterion_GAN(pred_real, valid)
            # Fake loss
            pred_fake = discriminator2(fake_sim_HR.detach(), real_sim_LR)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D2 = 0.5 * (loss_real + loss_fake)
            loss_D = 0.5 * (loss_D + loss_D2)

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] [S loss: %f] ETA: %s"
            % (
                epoch + 1,
                conf.n_epochs,
                i + 1,
                n_batches,
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item() * lambda_pixel,
                loss_GAN.item() * gamma_pixel,
                loss_saliency.item() * beta_pixel,
                time_left,
            )
        )

        # Record values
        report.record(loss_D.item(),
                      loss_G.item(),
                      loss_pixel.item() * lambda_pixel,
                      loss_GAN.item() * gamma_pixel,
                      loss_saliency.item() * beta_pixel)

    # =========================================================================
    # Main train loop
    # =========================================================================
    for epoch in range(conf.epoch, conf.n_epochs):
        for i in range(n_batches):
            train(i, time_left, epoch)

        report.push('train')

        for j in range(n_batches_val):
            validate(j, time_left, epoch)

        report.push('val')

        # After epoch, update schedule
        if conf.scheduler:
            scheduler_D.step()
            scheduler_G.step()
            if conf.dual_D:
                scheduler_D2.step()

        # Save images
        sample_images(epoch)
        sample_images_real(epoch)

        # Plot
        xx = np.linspace(1, len(report.G_loss_train), len(report.G_loss_train))
        plt.plot(xx, np.log(np.array(report.D_loss_train)),
                 xx, np.log(np.array(report.G_loss_train)),
                 xx, np.log(np.array(report.pix_loss_train)), ':g',
                 xx, np.log(np.array(report.adv_loss_train)), ':r',
                 xx, np.log(np.array(report.sal_loss_train)), 'b')
        plt.legend(['D', 'G', 'pix', 'adv', 'Sal'])
        plt.draw()
        plt.pause(0.001)

        xx = np.linspace(1, len(report.G_loss_val), len(report.G_loss_val))
        plt.plot(xx, np.log(np.array(report.D_loss_train)), 'b',
                 xx, np.log(np.array(report.G_loss_train)), 'r',
                 xx, np.log(np.array(report.D_loss_val)), ':b',
                 xx, np.log(np.array(report.G_loss_val)), ':r')
        plt.legend(['D', 'G', 'D val', 'G val'])
        plt.draw()
        plt.pause(0.001)

        # Determine approximate time left
        epoch_left = conf.n_epochs - epoch
        time_left = datetime.timedelta(
            seconds=epoch_left * (time.time() - prev_time) / (epoch + 1)
        )

        if ((conf.checkpoint_interval != -1 and epoch % conf.checkpoint_interval == 0) or
                epoch == conf.n_epochs - 1):
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       os.path.join(modeldir, 'generator_%d.pth' % epoch))
            torch.save(discriminator.state_dict(),
                       os.path.join(modeldir, 'discriminator_%d.pth' % epoch))
            if conf.dual_D:
                torch.save(discriminator2.state_dict(),
                           os.path.join(modeldir, 'discriminator2_%d.pth' % epoch))

    # Save loss plot
    xx = np.linspace(1, len(report.G_loss_train), len(report.G_loss_train))
    fig, ax = plt.subplots()
    plt.plot(xx, np.log(np.array(report.D_loss_train)),
             xx, np.log(np.array(report.G_loss_train)),
             xx, np.log(np.array(report.pix_loss_train)), ':g',
             xx, np.log(np.array(report.adv_loss_train)), ':r',
             xx, np.log(np.array(report.sal_loss_train)), 'b')
    plt.legend(['D', 'G', 'pix', 'adv', 'Sal'])
    plt.draw()
    plt.pause(0.001)
    fig.savefig(os.path.join(imagedir, 'Losses.png'))

    xx = np.linspace(1, len(report.G_loss_val), len(report.G_loss_val))
    fig, ax = plt.subplots()
    plt.plot(xx, np.log(np.array(report.D_loss_train)), 'b',
             xx, np.log(np.array(report.G_loss_train)), 'r',
             xx, np.log(np.array(report.D_loss_val)), ':b',
             xx, np.log(np.array(report.G_loss_val)), ':r')
    plt.legend(['D', 'G', 'D val', 'G val'])
    plt.draw()
    plt.pause(0.001)
    fig.savefig(os.path.join(imagedir, 'LossesVal.png'))

    xx = np.linspace(1, len(report.G_loss_train), len(report.G_loss_train))
    fig, ax = plt.subplots()
    plt.plot(xx, np.log(np.array(report.pix_loss_train)), 'r',
             xx, np.log(np.array(report.adv_loss_train)), 'g',
             xx, np.log(np.array(report.sal_loss_train)), 'b',
             xx, np.log(np.array(report.pix_loss_val)), ':r',
             xx, np.log(np.array(report.adv_loss_val)), ':g',
             xx, np.log(np.array(report.sal_loss_val)), ':b')
    plt.legend(['pix', 'adv', 'sal', 'pix-val', 'adv-val', 'sal-val'])
    plt.draw()
    plt.pause(0.001)

    # Save raw
    with open(os.path.join(imagedir, 'Losses.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(report.D_loss_train)
        wr.writerow(report.G_loss_train)
        wr.writerow(report.pix_loss_train)
        wr.writerow(report.adv_loss_train)
        wr.writerow(report.sal_loss_train)
        wr.writerow(report.D_loss_val)
        wr.writerow(report.G_loss_val)
        wr.writerow(report.pix_loss_val)
        wr.writerow(report.adv_loss_val)
        wr.writerow(report.sal_loss_val)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # NOTE: uses parent modules
    # from beam import    will not work unless modules are loaded or the module
    # is run as a script with the '-m' modifier
    conf = Config()
    train_model(conf, 'E:\\LSM-deeplearning\\20201027_Train_test')