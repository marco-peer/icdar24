from fnmatch import translate
import random
import numpy as np
from PIL import Image
import doxapy
from kornia import morphology as morph
import torch
import torchvision.transforms as transforms

class Binarize:

    def binarize(self, img):
        to_b = np.array(img.convert('L'))
        doxapy.Binarization.update_to_binary(doxapy.Binarization.Algorithms.SU, to_b)
        return Image.fromarray(to_b)

    def __call__(self, img):
        return self.binarize(img)



def get_random_kernel():
    k = torch.rand(3,3).round()#.cuda()
    k[1,1] = 1
    return k


class Erosion:
    def __init__(self):
        self.fn = morph.erosion

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class Dilation:
    def __init__(self):
        self.fn = morph.dilation

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]
    
class RandomDilation(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fn = Dilation()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
    
class RandomErosion(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fn = Erosion()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


import torchvision.transforms as transforms

def get_page_transforms(args):
    tfs = []

    tfs.append(transforms.RandomCrop(args['page_transforms'].get('crop_size', 512), pad_if_needed=True, fill=0))
    
    if p := args['page_transforms'].get('grayscale', 0.): tfs.append(transforms.RandomGrayscale(p=p))
    if p := args['page_transforms'].get('binarize', 0.): tfs.append(transforms.RandomApply([Binarize()], p = 0.2))
    return transforms.Compose(tfs)

def get_patch_transforms(args):
    tfs = []
    
    if args['patch_transforms'].get('jitter', None):
        tfs.append(transforms.ColorJitter(**args['patch_transforms']['jitter']))

    if p := args['patch_transforms'].get('erosion', None): tfs.append(RandomErosion(p))
    if p := args['patch_transforms'].get('dilation', None): tfs.append(RandomDilation(p))

    if args['patch_transforms'].get('affine', None):
        tfs.append(transforms.RandomAffine(**args['patch_transforms']['affine'], fill=0))

    tfs.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(tfs)
