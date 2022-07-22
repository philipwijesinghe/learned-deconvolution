# -*- coding: utf-8 -*-
""" Reported Class stores model loss parameters for logging
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2022 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)


def mean(data):
    return sum(data) / len(data)


class Reporter:
    def __init__(self):
        """Constructs a class with many list props for logging
        """
        # Recorded losses per epoch
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
        # Temporary lists for batch loop
        self.G_loss = []
        self.D_loss = []
        self.pix_loss = []
        self.adv_loss = []
        self.sal_loss = []

    def record(self, dloss, gloss, pix, adv, sal):
        """Records losses per batch and appends to temporary list

        Parameters
        ----------
        dloss
        gloss
        pix
        adv
        sal
        """
        self.D_loss.append(dloss)
        self.G_loss.append(gloss)
        self.pix_loss.append(pix)
        self.adv_loss.append(adv)
        self.sal_loss.append(sal)

    def push(self, where):
        """Pushes recorded temp lists to permanent record based on train or val run

        Parameters
        ----------
        where
            'train' or 'val'
        """
        if where == 'train':
            self.D_loss_train.append(mean(self.D_loss))
            self.G_loss_train.append(mean(self.G_loss))
            self.pix_loss_train.append(mean(self.pix_loss))
            self.adv_loss_train.append(mean(self.adv_loss))
            self.sal_loss_train.append(mean(self.sal_loss))
        elif where == 'val':
            self.D_loss_val.append(mean(self.D_loss))
            self.G_loss_val.append(mean(self.G_loss))
            self.pix_loss_val.append(mean(self.pix_loss))
            self.adv_loss_val.append(mean(self.adv_loss))
            self.sal_loss_val.append(mean(self.sal_loss))

        self.G_loss = []
        self.D_loss = []
        self.pix_loss = []
        self.adv_loss = []
        self.sal_loss = []

    # def normalize(self):
    #     self.D_lossN = self.D_loss / np.max(self.D_loss)
    #     self.G_lossN = self.G_loss / np.max(self.G_loss)
    #     self.pix_lossN = self.pix_loss / np.max(self.pix_loss)
    #     self.adv_lossN = self.adv_loss / np.max(self.adv_loss)
    #     self.sal_lossN = self.sal_loss / np.max(self.sal_loss)
