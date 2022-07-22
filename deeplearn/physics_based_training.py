# -*- coding: utf-8 -*-
""" Trains a physics-based model
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import sys
sys.path.append('../')

import os
import time
import datetime
import csv
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from deeplearn.models import GeneratorResNet, GeneratorUNet64US
from deeplearn.models import Discriminator, DiscriminatorSN, weights_init_normal
from deeplearn.datasets import ImageDatasetLSM, ImageDatasetRef
from deeplearn.perceptual_loss import VGGPerceptualLoss

from deeplearn.helper import Reporter

from fileio.yamlio import load_config

# This isn't needed here, but other functions may import from this module (backwards compatibility)
# TODO: fix dependencies
from deeplearn.config import Config


# TODO: this should be converted to a separate training and viewer class, and then controlled via a `semantic'
# controller class
def train_model(conf, dataroot):
    """Trains a physics-based model for deconvolution based on config

    Structure:
    First the network architecture and dataloaders are initialised based on configs
    Then single batch training/validation functions are defined (as nested functions)
    Finally, a training loop iterates the train/validation functions and saves outputs

    Parameters
    ----------
    conf
        Config Class object from config.py with user configuration
    dataroot
        path to folder that contains the training images (and TrainConfig.yml file)

    """
    # config, _, _ = load_config(datadir + '/TrainConfig.yml')

    # GPU environment
    cuda = True if torch.cuda.is_available() else False
    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Prepare folders
    imagedir_sim = os.path.join(dataroot, 'training_images_sim')
    imagedir_real = os.path.join(dataroot, 'training_images_real')
    modeldir = os.path.join(dataroot, 'saved_models')
    os.makedirs(imagedir_sim, exist_ok=True)
    os.makedirs(imagedir_real, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)

    # Loss functions
    criterion_gan = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_saliency = VGGPerceptualLoss()

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, conf.img_size // 2 ** 4, conf.img_size // 2 ** 4)

    # Initialize generator and discriminator
    # TODO: replace with arbitrary model load from name and separate models into subfolder
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__"))
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
    else:
        print("No model for image size specified. Currently limited to 64 pix width.")
        return

    # Defines one or two discriminators with or withour spectral normalisation
    if conf.spectral_norm:
        discriminator = DiscriminatorSN(in_channels=conf.channels)
        discriminator2 = DiscriminatorSN(in_channels=conf.channels) if conf.dual_D else None
    else:
        discriminator = Discriminator(in_channels=conf.channels)
        discriminator2 = Discriminator(in_channels=conf.channels) if conf.dual_D else None

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_gan.cuda()
        criterion_pixelwise.cuda()
        criterion_saliency.cuda()
        if conf.dual_D:
            discriminator2.cuda()

    # Init start state
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
    optimizer_g = torch.optim.Adam(generator.parameters(),
                                   lr=conf.lrG,
                                   betas=(conf.b1, conf.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                   lr=conf.lrD,
                                   betas=(conf.b1, conf.b2))
    optimizer_d2 = torch.optim.Adam(discriminator2.parameters(),
                                    lr=conf.lrD,
                                    betas=(conf.b1, conf.b2)) if conf.dual_D else None

    # Rate schedule
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_g, milestones=[150, 250], gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d, milestones=[150, 250], gamma=0.1)
    scheduler_d2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d2, milestones=[150, 250], gamma=0.1) if conf.dual_D else None

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
    )  # Mini dataloader for visualising performance during training

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

    # =========================================================================
    # Prepare training
    # =========================================================================
    prev_time = time.time()
    time_left = 0

    report = Reporter()
    plt.subplots()

    # Preloads all the images into RAM - this is much faster
    # Increases RAM usage, but ok since we typically use ~1000 64x64 images
    full_batch = next(iter(dataloader_train))
    real_batch = next(iter(dataloader_real))
    val_batch = next(iter(dataloader_val))
    val_real_batch = next(iter(dataloader_val_real))

    n_batches = (len(dataset_train) // conf.batch_size)
    n_batches_real = (len(dataset_real) // conf.batch_size)
    n_batches_val = (len(dataset_val) // conf.batch_size)
    n_batches_val_real = (len(dataset_val_real) // conf.batch_size)

    # =============================================================================
    # Visualisation
    # =============================================================================
    def sample_images_sim(batches_done):
        """Saves a generated sample from the simulated validation set to an image"""
        imgs = next(iter(dataloader_val_display))
        real_lr = Variable(imgs["LR"].type(tensor))
        real_hr = Variable(imgs["HR"].type(tensor))
        fake_hr = generator(real_lr)
        img_sample = torch.cat((real_hr.data, fake_hr.data, real_lr.data), -2)
        save_image(img_sample,
                   imagedir_sim + "/%s.png" % batches_done,
                   nrow=6,
                   normalize=True)

    def sample_images_real(batches_done):
        """Saves a generated sample from the real validation set to an image"""
        imgs = next(iter(dataloader_val_real_display))
        real_lr = Variable(imgs["REF"].type(tensor))
        fake_hr = generator(real_lr)
        img_sample = torch.cat((real_lr.data, fake_hr.data), -2)
        save_image(img_sample,
                   imagedir_real + "/%s.png" % batches_done,
                   nrow=6,
                   normalize=True)

    # =========================================================================
    # Training functions
    # =========================================================================
    def train(batch_no, epoch_no, time_left_instance):
        # Load batch manually from full data in RAM
        batch_lr = full_batch["LR"][batch_no * conf.batch_size:
                                    min((batch_no + 1) * conf.batch_size, len(dataset_train))]
        batch_hr = full_batch["HR"][batch_no * conf.batch_size:
                                    min((batch_no + 1) * conf.batch_size, len(dataset_train))]

        # there are often fewer real images than simulated
        # therefore, we reuse real images, so they equally weigh towards training
        batch_no_real = batch_no % n_batches_real
        batch_real = real_batch["REF"][batch_no_real * conf.batch_size:
                                       min((batch_no_real + 1) * conf.batch_size, len(dataset_real))]

        # Model inputs
        in_sim_lr = Variable(batch_lr.type(tensor))
        in_sim_hr = Variable(batch_hr.type(tensor))
        in_real_lr = Variable(batch_real.type(tensor))

        # Adversarial ground truth labels
        # Smooth one-sided labeling (0.9 vs 1) minimises oversaturation (overconfidence) of discriminator.
        # See: Salimans, T. et al. Improved Techniques for Training GANs. arXiv:1606.03498 [cs] (2016).
        valid_label = 0.9 if conf.smooth_labels else 1
        fake_label = 0  # generally one-sided smooth label is best
        valid = Variable(tensor(valid_label * np.ones((in_sim_lr.size(0), *patch))), requires_grad=False)
        fake = Variable(tensor(np.zeros((in_sim_lr.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_g.zero_grad()

        out_sim_hr = generator(in_sim_lr)
        out_real = generator(in_real_lr)

        # Losses
        pred_fake = discriminator(out_sim_hr, in_sim_lr)
        loss_gan = criterion_gan(pred_fake, valid)
        if conf.dual_D:
            # A generator may overfit to specifically trick a discriminator; locking them in a min-max game which may
            # have clear image artifacts. This happens when the adversarial loss heavily weighs in to train a
            # generator. A second discriminator that is not used to train the generator will likely pick up on these
            # 'non-physical' image features very quickly. Here, we select the discriminator with the best
            # discrimination capacity to train the generator. If they get lockes, the 'worse' discriminator will
            # become 'better' and swap out.
            pred_fake2 = discriminator2(out_sim_hr, in_sim_lr)
            loss_gan2 = criterion_gan(pred_fake2, valid)
            loss_gan = torch.max(loss_gan, loss_gan2)

        loss_pixel = criterion_pixelwise(out_sim_hr, in_sim_hr)
        loss_saliency = criterion_saliency(out_real, in_real_lr)

        # Backpropagate losses
        # TODO: replace names with semantics
        loss_g = (conf.gamma_pixel * loss_gan +
                  conf.lambda_pixel * loss_pixel +
                  conf.beta_pixel * loss_saliency)
        loss_g.backward()
        optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_d.zero_grad()

        pred_real = discriminator(in_sim_hr, in_sim_lr)
        loss_real = criterion_gan(pred_real, valid)
        pred_fake = discriminator(out_sim_hr.detach(), in_sim_lr)
        loss_fake = criterion_gan(pred_fake, fake)

        loss_d = 0.5 * (loss_real + loss_fake)
        loss_d.backward()
        optimizer_d.step()

        if conf.dual_D:
            optimizer_d2.zero_grad()

            pred_real = discriminator2(in_sim_hr, in_sim_lr)
            loss_real = criterion_gan(pred_real, valid)
            pred_fake = discriminator2(out_sim_hr.detach(), in_sim_lr)
            loss_fake = criterion_gan(pred_fake, fake)

            loss_d2 = 0.5 * (loss_real + loss_fake)
            loss_d2.backward()
            optimizer_d2.step()

            loss_d = 0.5 * (loss_d + loss_d2)  # average loss for records

        # ---------------------
        #  Records
        # ---------------------
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] [S loss: %f] ETA: %s"
            % (
                epoch_no + 1,
                conf.n_epochs,
                batch_no + 1,
                n_batches,
                loss_d.item(),
                loss_g.item(),
                loss_pixel.item() * conf.lambda_pixel,
                loss_gan.item() * conf.gamma_pixel,
                loss_saliency.item() * conf.beta_pixel,
                time_left_instance,
            )
        )

        report.record(loss_d.item(),
                      loss_g.item(),
                      loss_pixel.item() * conf.lambda_pixel,
                      loss_gan.item() * conf.gamma_pixel,
                      loss_saliency.item() * conf.beta_pixel)

    def validate(batch_no, epoch_no, time_left_instance):
        # Load batch manually from full data in RAM
        batch_lr = val_batch["LR"][batch_no * conf.batch_size:
                                   min((batch_no + 1) * conf.batch_size, len(dataset_val))]
        batch_hr = val_batch["HR"][batch_no * conf.batch_size:
                                   min((batch_no + 1) * conf.batch_size, len(dataset_val))]
        # there are often fewer real images than simulated
        # therefore, we reuse real images, so they equally weigh towards training
        batch_no_real = batch_no % n_batches_val_real
        batch_real = val_real_batch["REF"][batch_no_real * conf.batch_size:
                                           min((batch_no_real + 1) * conf.batch_size, len(dataset_val_real))]

        # Model inputs
        in_sim_lr = Variable(batch_lr.type(tensor))
        in_sim_hr = Variable(batch_hr.type(tensor))
        in_real = Variable(batch_real.type(tensor))

        # Adversarial ground truth labels
        # Smooth one-sided labeling (0.9 vs 1) minimises oversaturation (overconfidence) of discriminator.
        # See: Salimans, T. et al. Improved Techniques for Training GANs. arXiv:1606.03498 [cs] (2016).
        valid_label = 0.9 if conf.smooth_labels else 1
        fake_label = 0  # generally one-sided smooth label is best
        valid = Variable(tensor(valid_label * np.ones((in_sim_lr.size(0), *patch))), requires_grad=False)
        fake = Variable(tensor(np.zeros((in_sim_lr.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Validate Generator
        # ------------------
        out_sim_hr = generator(in_sim_lr)
        out_real = generator(in_real)

        # Losses
        pred_fake = discriminator(out_sim_hr, in_sim_lr)
        loss_gan = criterion_gan(pred_fake, valid)
        if conf.dual_D:
            pred_fake2 = discriminator2(out_sim_hr, in_sim_lr)
            loss_gan2 = criterion_gan(pred_fake2, valid)
            loss_gan = torch.min(loss_gan, loss_gan2)

        loss_pixel = criterion_pixelwise(out_sim_hr, in_sim_hr)
        loss_saliency = criterion_saliency(out_real, in_real)

        # Record losses
        loss_g = (conf.gamma_pixel * loss_gan +
                  conf.lambda_pixel * loss_pixel +
                  conf.beta_pixel * loss_saliency)

        # ---------------------
        #  Validate Discriminator
        # ---------------------
        pred_real = discriminator(in_sim_hr, in_sim_lr)
        loss_real = criterion_gan(pred_real, valid)
        pred_fake = discriminator(out_sim_hr.detach(), in_sim_lr)
        loss_fake = criterion_gan(pred_fake, fake)
        loss_d = 0.5 * (loss_real + loss_fake)

        if conf.dual_D:
            pred_real = discriminator2(in_sim_hr, in_sim_lr)
            loss_real = criterion_gan(pred_real, valid)
            pred_fake = discriminator2(out_sim_hr.detach(), in_sim_lr)
            loss_fake = criterion_gan(pred_fake, fake)
            loss_d2 = 0.5 * (loss_real + loss_fake)
            loss_d = 0.5 * (loss_d + loss_d2)

        # ---------------------
        #  Records
        # ---------------------
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] [S loss: %f] ETA: %s"
            % (
                epoch_no + 1,
                conf.n_epochs,
                batch_no + 1,
                n_batches,
                loss_d.item(),
                loss_g.item(),
                loss_pixel.item() * conf.lambda_pixel,
                loss_gan.item() * conf.gamma_pixel,
                loss_saliency.item() * conf.beta_pixel,
                time_left_instance,
            )
        )

        report.record(loss_d.item(),
                      loss_g.item(),
                      loss_pixel.item() * conf.lambda_pixel,
                      loss_gan.item() * conf.gamma_pixel,
                      loss_saliency.item() * conf.beta_pixel)

    # =========================================================================
    # Main train loop
    # =========================================================================
    for epoch in range(conf.epoch, conf.n_epochs):
        for i in range(n_batches):
            train(i, epoch, time_left)
        report.push('train')

        for j in range(n_batches_val):
            validate(j, epoch, time_left)
        report.push('val')

        # After epoch, update schedule
        if conf.scheduler:
            scheduler_d.step()
            scheduler_g.step()
            if conf.dual_D:
                scheduler_d2.step()

        # Save images
        sample_images_sim(epoch)
        sample_images_real(epoch)

        # TODO: Visualisation should be a separate 'View' Class
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
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - prev_time) / (epoch + 1))

        if (conf.checkpoint_interval != -1 and epoch % conf.checkpoint_interval == 0) or epoch == conf.n_epochs - 1:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(modeldir, 'generator_%d.pth' % epoch))
            torch.save(discriminator.state_dict(), os.path.join(modeldir, 'discriminator_%d.pth' % epoch))
            if conf.dual_D:
                torch.save(discriminator2.state_dict(), os.path.join(modeldir, 'discriminator2_%d.pth' % epoch))

    # TRAINING COMPLETED
    # Save final loss plots
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
    fig.savefig(os.path.join(imagedir_sim, 'Losses.png'))

    xx = np.linspace(1, len(report.G_loss_val), len(report.G_loss_val))
    fig, ax = plt.subplots()
    plt.plot(xx, np.log(np.array(report.D_loss_train)), 'b',
             xx, np.log(np.array(report.G_loss_train)), 'r',
             xx, np.log(np.array(report.D_loss_val)), ':b',
             xx, np.log(np.array(report.G_loss_val)), ':r')
    plt.legend(['D', 'G', 'D val', 'G val'])
    plt.draw()
    plt.pause(0.001)
    fig.savefig(os.path.join(imagedir_sim, 'LossesVal.png'))

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
    with open(os.path.join(imagedir_sim, 'Losses.csv'), 'w', newline='') as myfile:
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

# # =============================================================================
# # MAIN
# # =============================================================================
# if __name__ == "__main__":
#     # Run this section as a script, while making all defined functions
#     # available to the module
#
#     # NOTE: uses parent modules
#     # from beam import    will not work unless modules are loaded or the module
#     # is run as a script with the '-m' modifier
#     conf = Config()
#     train_model(conf, 'E:\\LSM-deeplearning\\20201027_Train_test')
