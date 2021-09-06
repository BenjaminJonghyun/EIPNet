"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--weight_path', type=str, default='checkpoint/CelebA', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--image_save_path', type=str, default='result')

        parser.add_argument('--status', type=str, default='test')

        self.isTrain = False
        return parser
