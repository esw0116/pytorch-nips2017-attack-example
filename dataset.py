import os
import glob
import os.path
import torch
import pandas as pd

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg']


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def default_inception_transform(img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf


def find_inputs(folder, filename_to_target=None, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                target = filename_to_target[rel_filename] if filename_to_target else 0
                inputs.append((abs_filename, target))
    return inputs


class Dataset(data.Dataset):

    def __init__(self, args, dir_data=None, class_file='images.csv', transform=None):
        self.args = args
        if dir_data is None:
            dir_data = args.input_dir
        self.transform = transform
        class_df = pd.read_csv(os.path.join(args.input_dir, class_file), index_col='ImageId')
        self.true_class = dict(class_df.TrueLabel)
        self.target_class = dict(class_df.TargetClass)
        self.imgs = glob.glob(os.path.join(dir_data, '**/*.png'), recursive=True)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + dir_data + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img_name = os.path.splitext(os.path.basename(self.imgs[index]))[0]
        target = self.target_class[img_name] - 1
        true_cl = self.true_class[img_name] - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, target, true_cl, img_name

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def filenames(self, indices=[]):
        if indices:
            return [self.imgs[i] for i in indices]
        else:
            return [x for x in self.imgs]
