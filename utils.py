import cv2
import os
import numpy as np
import random
import glob

def make_data_path(train_list, test_list):
    train = glob.glob(os.path.join(train_list, '*'))
    test = glob.glob(os.path.join(test_list, '*'))
    return train, test


class data_augmentation(object):
    def __init__(self, train_list, test_list):

        self.train_list = train_list
        self.test_list = test_list

    def open_image(self, path, width, height, isCrop=False, isResize=True, isflip=False, isRotate=False, angle=0):
        # print(self.data_list[class_list][subclass_list][start])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isCrop:
            img = img[20:198, 0:178]
        if isResize:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        if isflip:  # horizontal flip
            img = cv2.flip(img, 1)
        if isRotate:
            img_waf = img
            img = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            img = cv2.warpAffine(img_waf, img,(width, height))

        img_lr_2 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        img_lr_4 = cv2.resize(img_lr_2, (32, 32), interpolation=cv2.INTER_LINEAR)
        img_lr = cv2.resize(img_lr_4, (16, 16), interpolation=cv2.INTER_LINEAR)

        img_grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grad = cv2.Canny(img_grad, 50, 255)

        img_grad_2 = cv2.cvtColor(img_lr_2, cv2.COLOR_BGR2GRAY)
        img_grad_2 = cv2.Canny(img_grad_2, 50, 255)

        img_grad_4 = cv2.cvtColor(img_lr_4, cv2.COLOR_BGR2GRAY)
        img_grad_4 = cv2.Canny(img_grad_4, 50, 255)

        img = img.astype(np.float32)
        img_lr = img_lr.astype(np.float32)
        img_grad = img_grad.astype(np.float32)
        img_grad = np.reshape(img_grad, (128, 128, 1))
        img_grad_2 = img_grad_2.astype(np.float32)
        img_grad_2 = np.reshape(img_grad_2, (64, 64, 1))
        img_grad_4 = img_grad_4.astype(np.float32)
        img_grad_4 = np.reshape(img_grad_4, (32, 32, 1))

        return img, img_lr, img_grad, img_grad_2, img_grad_4

    def next_batch(self, batch_size, width, height):
        imgs = []
        imgs_lr = []
        imgs_grad = []
        imgs_grad_2 = []
        imgs_grad_4 = []

        for i in range(0, batch_size):
            train_img = random.choice(self.train_list)

            flip = random.choice([True, False])
            rotation = random.choice([True, False])
            angle = random.choice([90, 180, 270])

            img, img_lr, img_grad, img_grad_2, img_grad_4 = self.open_image(train_img, width=width,
                                                                            height=height, isflip=flip,
                                                                            isRotate=rotation, angle=angle)
            imgs.append(img)
            imgs_lr.append(img_lr)
            imgs_grad.append(img_grad)
            imgs_grad_2.append(img_grad_2)
            imgs_grad_4.append(img_grad_4)

        imgs = np.array(imgs)
        imgs_lr = np.array(imgs_lr)
        imgs_grad = np.array(imgs_grad)
        imgs_grad_2 = np.array(imgs_grad_2)
        imgs_grad_4 = np.array(imgs_grad_4)

        return imgs, imgs_lr, imgs_grad, imgs_grad_2, imgs_grad_4

    def next_batch_test(self, batch_size, width, height):
        imgs = []
        imgs_lr = []
        imgs_grad = []
        imgs_grad_2 = []
        imgs_grad_4 = []

        for i in range(0, batch_size):
            test_img = random.choice(self.test_list)

            flip = False
            rotation = False
            angle = 0

            img, img_lr, img_grad, img_grad_2, img_grad_4 = self.open_image(test_img, width=width,
                                                                            height=height, isflip=flip,
                                                                            isRotate=rotation, angle=angle)
            imgs.append(img)
            imgs_lr.append(img_lr)
            imgs_grad.append(img_grad)
            imgs_grad_2.append(img_grad_2)
            imgs_grad_4.append(img_grad_4)

        imgs = np.array(imgs)
        imgs_lr = np.array(imgs_lr)
        imgs_grad = np.array(imgs_grad)
        imgs_grad_2 = np.array(imgs_grad_2)
        imgs_grad_4 = np.array(imgs_grad_4)

        return imgs, imgs_lr, imgs_grad, imgs_grad_2, imgs_grad_4