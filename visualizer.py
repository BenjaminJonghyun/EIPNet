import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import os
import time

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoint_path, opt.save_path, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def loss_initialization(self):
        MSE = 0
        EDGE = 0
        YUV = 0
        G_GAN = 0
        D_GAN = 0
        G_total = 0
        losses = OrderedDict([('G_total', G_total),
                              ('MSE', MSE),
                              ('EDGE', EDGE),
                              ('YUV', YUV),
                              ('G_GAN', G_GAN),
                              ('D_GAN', D_GAN)])
        return losses


    def print_save_current_error(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.4f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def save_image(self, epoch, image_numpy, create_dir=True):
        #self.opt.isTrain = False
        if self.opt.isTrain:
            image_path = os.path.join(self.opt.checkpoint_path, self.opt.save_path, 'images')
        else:
            image_path = os.path.join(self.opt.checkpoint_path, self.opt.save_path, 'test_images')
        if create_dir:
            if not os.path.exists(image_path):
                os.makedirs(image_path)
        for k, v in image_numpy.items():
            image = self.tile_image(v)
            image_pil = Image.fromarray(image.astype(np.uint8))
            if self.opt.isTrain:
                image_name = '%d_epoch_%s_image.png' % (epoch, k)
            else:
                image_name = '%05d_%s.png' % (epoch, k)
            image_pil.save(os.path.join(image_path, image_name))


    def tile_image(self, array, num_tile = 4):
        b, h, w, c = np.shape(array)
        h_size = int(b / num_tile)
        if h_size > 0:
            count = 0
            new_array = np.zeros((h * h_size, w * num_tile, c))
            for h_s in range(h_size):
                for w_s in range(num_tile):
                    new_array[h_s*h:(h_s+1)*h, w_s*w:(w_s+1)*w, :] = array[count]
                    count += 1
        else:
            new_array = array[0]
        return np.array(new_array)


