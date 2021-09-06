"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for save and display
        parser.add_argument('--display_epoch_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--print_epoch_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')

        # for training
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--ad_r', type=float, default=5e-3)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=int,  help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--max_to_keep', type=int, default=10, help='')

        # for discriminators
        parser.add_argument('--lambda_edge', type=float, default=0.3, help='weight for feature matching loss')
        parser.add_argument('--status', type=str, default='train')

        self.isTrain = True
        return parser
