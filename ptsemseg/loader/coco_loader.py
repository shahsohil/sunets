import os
import random
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageMath
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from ptsemseg import get_data_path

# These arrays are used for mapping between coco and pascal voc classes
pascal_classes = np.array([1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72])
pascal_map = np.array([  0.,  15.,   2.,   7.,  14.,   1.,   6.,  19.,   0.,   4.,   0.,
         0.,   0.,   0.,   0.,   0.,   3.,   8.,  12.,  13.,  17.,  10.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         5.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   9.,  18.,  16.,   0.,
         0.,  11.,   0.,   0.,   0.,   0.,  20.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0., 91], dtype=np.int32)

class COCOLoader(data.Dataset):
    def __init__(self, root, split="train_aug", is_transform=False, img_size=512):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.ignore_index = 91
        self.n_classes = 21
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = collections.defaultdict(list)

        self.image_transform = Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.filler = [0, 0, 0]

        # Reading COCO dataset list - train2014, val2014, train_aug, val, test2014, test2015
        self.data_path = get_data_path('coco')
        filepath = self.data_path + '/annotations/' + split + '.txt', 'r'
        if split is "train_aug" and not os.path.exists(filepath):
            self.filtertraindata()
        file_list = tuple(open(filepath))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_name = self.files[index]
        img, lbl = self.readfile(img_name)

        if 'val' in self.split or 'test' in self.split:
            img, lbl = self.r_crop(img, lbl)
        elif self.is_transform:
            img, lbl = self.transform(img, lbl)

        # mean subtraction
        img = self.image_transform(img)

        lbl = np.array(lbl, dtype=np.int32)

        # mapping of lbl to pascal labels
        lbl = pascal_map[lbl]

        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def readfile(self, img_name):

        img_path = self.data_path + img_name + '.jpg'
        lbl_path = self.data_path + 'seg_mask' + img_name + '.png'

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert('P')

        return img, lbl

    def transform(self, img, lbl):
        # Scaling
        img, lbl = self.r_scale(img, lbl)

        # Cropping
        img, lbl = self.r_crop(img, lbl)

        # Flipping
        img, lbl = self.r_flip(img, lbl)

        # Rotation
        img, lbl = self.r_rotate(img, lbl)

        return img, lbl

    def r_scale(self, img, lbl, low=0.5, high=2.0):
        w, h = img.size

        resize = random.uniform(low, high)

        new_w, new_h = int(resize * w), int(resize * h)

        image_transform = Resize(size=(new_h, new_w))
        label_transform = Resize(size=(new_h, new_w), interpolation=Image.NEAREST)

        return (image_transform(img), label_transform(lbl))

    def r_crop(self, img, lbl):
        w, h = img.size
        th, tw = self.img_size
        if w < tw or h < th:
            padw, padh = max(tw - w, 0), max(th - h, 0)
            w += padw
            h += padh
            im = Image.new(img.mode, (w, h), tuple(self.filler))
            im.paste(img, (int(padw / 2), int(padh / 2)))
            l = Image.new(lbl.mode, (w, h), self.ignore_index)
            l.paste(lbl, (int(padw / 2), int(padh / 2)))
            img = im
            lbl = l
        if w == tw and h == th:
            return img, lbl
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), lbl.crop((x1, y1, x1 + tw, y1 + th)))

    def r_flip(self, img, lbl):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl

    def r_rotate(self, img, lbl):
        angle = random.uniform(-10, 10)

        lbl = np.array(lbl, dtype=np.int32) - self.ignore_index
        lbl = Image.fromarray(lbl)
        img = tuple([ImageMath.eval("int(a)-b", a=j, b=self.filler[i]) for i, j in enumerate(img.split())])

        lbl = lbl.rotate(angle, resample=Image.NEAREST)
        img = tuple([k.rotate(angle, resample=Image.BICUBIC) for k in img])

        lbl = ImageMath.eval("int(a)+b", a=lbl, b=self.ignore_index)
        img = Image.merge(mode='RGB', bands=tuple(
            [ImageMath.eval("convert(int(a)+b,'L')", a=j, b=self.filler[i]) for i, j in enumerate(img)]))
        return (img, lbl)

    def get_pascal_labels(self):
        return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                           [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                           [0, 192, 0], [128, 192, 0], [0, 64, 128]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.all(mask == np.array(label).reshape(1, 1, 3), axis=2)] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    # Function used for filtering coco dataset and creating train_aug and val.txt file for pascal pretraining
    def filtertraindata(self):
        datapath = get_data_path('coco')
        train_list = tuple(open(datapath + 'annotations/train2014.txt', 'r'))
        val_list = tuple(open(datapath + 'annotations/val2014.txt', 'r'))
        total_list = ['/train2014/'+id_.rstrip() for id_ in train_list] + ['/val2014/'+id_.rstrip() for id_ in val_list]

        annotation_path = os.path.join(datapath, 'seg_mask')
        aug_list = []
        for filename in total_list:
            lbl_path = annotation_path + filename + '.png'
            lbl = Image.open(lbl_path).convert('P')
            lbl = np.array(lbl, dtype=np.int32)
            if np.sum(pascal_map[lbl] != 0) > 1000 and np.intersect1d(np.unique(lbl),pascal_classes).any():
                aug_list.append(filename)

        val_aug_list = random.sample(aug_list, 1500)
        train_aug_list = list(set(aug_list) - set(val_aug_list))
        with open(os.path.join(datapath, 'annotations', 'train_aug.txt'), 'w') as txtfile:
            [txtfile.write(file + '\n') for file in train_aug_list]
        with open(os.path.join(datapath, 'annotations', 'val.txt'), 'w') as txtfile:
            [txtfile.write(file + '\n') for file in val_aug_list]