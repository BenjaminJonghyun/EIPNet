"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NsC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os
import pickle


class BaseOptions():
    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--save_path', type=str, default='CelebA', help='name of the experiment. It decides where to store samples')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='models are saved here')
        parser.add_argument('--data_path', type=str, default='datasets', help='datasets are saved in here')

        # input/output
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=128, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=128, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.save_path)
        if makedir:
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def parse(self, save=False):

        opt = self.gather_options()
        self.save_options(opt)
        self.print_options(opt)
        self.opt = opt
        return self.opt

