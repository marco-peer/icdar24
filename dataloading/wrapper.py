from itertools import repeat
import math
import random
import cv2
import torch.utils.data as data
import doxapy
from PIL import Image
import os
import os.path
from tqdm import tqdm
import numpy as np
import re
from PIL import ImageOps
import glob
import torch
from torch.utils.data import DataLoader
import logging
from torchvision import transforms

class WrapableDataset(data.Dataset):

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def supported_classes():
        return {
                'SelectLabels': 'SelectLabels',
                'TransformImages' : 'TransformImages',
                'TrainingSampler': 'TrainingSampler',
                'EvalSIFTSampler' : 'EvalSIFTSampler',
                'SamplerWithBinary': 'SamplerWithBinary',
                'EvalSIFTColorSampler': 'EvalSIFTColorSampler',
                'TrainSIFTSampler' : 'TrainSIFTSampler'
                }

    def _get_wrapper_class_constructor(self, name):
        def wrapper(*args, **kw):
            c = self.supported_classes()[name]
            if type(c) == str:
                return globals()[c](self, *args, **kw)
            else:
                return c(self, *args, **kw)

        return wrapper

    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)

    def __getitem__(self, index):
        return self.get_image(index), self.get_label(index)


class DatasetWrapper(WrapableDataset):

    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)
        else:
            return getattr(self.dataset, attr)

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


class CombineLabels(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.packed_labels = np.array([np.asscalar(np.argwhere(np.all(self.unique_labels == l, axis=1))) for l in
                                       self.packed_labels]).astype('int32')
        self.unique_labels = np.unique(self.packed_labels, axis=0)

    def get_label(self, index):
        return self.packed_labels[index]


class SelectLabels(DatasetWrapper):
    def __init__(self, *args, label_names, **kwargs):
        super().__init__(*args, **kwargs)
        # requires dict {label_name : labels}
        self.label_names = label_names if type(label_names) == list else [label_names]
        self.packed_labels = np.stack([self.labels[l] for l in self.label_names], axis=1)

    def get_label(self, index):
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __getitem__(self, index):
        return self.dataset.get_image(index), self.get_label(index)


class TransformImages(DatasetWrapper):
    def __init__(self, *args, transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
 
    def get_image(self, index):
        img = self.dataset.get_image(index)
        img = self.transform(img)
        return img
    

class TrainingSampler(DatasetWrapper):

    def __init__(self, *args, page_transforms, patch_size, num_keypoints, num_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints

        self.patch_size = patch_size
        self.page_transforms = page_transforms
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def crop(self, img, x,y, crop_size=None):
        crop_size = crop_size or (self.patch_size, self.patch_size)
        return img.crop((int(x-crop_size[0]/2), int(y-crop_size[1]/2), int(x+crop_size[0]/2), int(y+crop_size[1]/2)))

    def check_crop(self, tensor_crop):
        if tensor_crop.shape[1:] != (self.patch_size, self.patch_size):
            return False
        
        if torch.sum(1-tensor_crop.round()) < 0.08*len(tensor_crop.ravel()):
            return False
        if torch.sum(1-tensor_crop.round()) > 0.8*len(tensor_crop.ravel()):
            return False
        return True
              
    def detect_sift(self, img):

        # binarize for mask if image is not already binary
        if img.mode != 'L':
            to_b = np.array(img.convert('L'))
            doxapy.Binarization.update_to_binary(doxapy.Binarization.Algorithms.ISAUVOLA , to_b)
        else:
            to_b = np.array(img)

        # detect 
        nfeatures = self.num_keypoints if self.num_keypoints != -1 else 0
        sift = cv2.SIFT_create(sigma=3.2, nfeatures=nfeatures, enable_precise_upscale=True)
        kp = sift.detect(to_b, None)

        return kp, Image.fromarray(to_b)

    
    def __getitem__(self, index):
        image = self.dataset.get_image(index)
        labels = self.get_label(index)

        img = self.page_transforms(image)
        kps, mask = self.detect_sift(img)

        img = img.convert('RGB') if img.mode == 'L' else img  # make sure crops are consistent with three channels
        
        # iterate and filter the bad ones
        crops = []
        for kp in kps:
            masked_crop = transforms.functional.to_tensor(self.crop(mask, kp.pt[0], kp.pt[1]))

            if not self.check_crop(masked_crop):
                continue

            crops.append(transforms.functional.to_tensor(self.crop(img, kp.pt[0], kp.pt[1])))

        if not crops:
            return None
        
        # normalize crops
        crops = self.normalize(torch.stack(crops))

        # sample crops if we have too many of them
        if len(crops) > self.num_samples:
            idx = torch.linspace(0, len(crops) -1, self.num_samples).int()
            crops = crops[idx]


        if labels:
            return crops, labels
        else:
            return crops


class EvalSIFTSampler(DatasetWrapper):

    def __init__(self, *args, patch_size, num_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

        self.patch_size = patch_size
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def crop(self, img, x,y, crop_size=None):
        crop_size = crop_size or (self.patch_size, self.patch_size)
        return img.crop((int(x-crop_size[0]/2), int(y-crop_size[1]/2), int(x+crop_size[0]/2), int(y+crop_size[1]/2)))

    def check_crop(self, tensor_crop):
        if tensor_crop.shape[1:] != (self.patch_size, self.patch_size):
            return False
        
        c = tensor_crop
        # white > 95 %
        if torch.sum(c == 1) < c.ravel().shape[0] * 0.99: #0.95
            return True
        
        # if torch.sum(c == 0) > c.ravel().shape[0] * 0.05:
        #     if torch.sum(c == 1) < c.ravel().shape[0] * 0.95:
        #         return True

        ######

        
        # if torch.sum(1-tensor_crop.round()) < 0.05*len(tensor_crop.ravel()):
        #     return False
        # if torch.sum(1-tensor_crop.round()) > 0.95*len(tensor_crop.ravel()):
        #     return False
        

        # if torch.sum(1-tensor_crop.round()) < 0.08*len(tensor_crop.ravel()):
        #     return False
        # if torch.sum(1-tensor_crop.round()) > 0.8*len(tensor_crop.ravel()):
        #     return False
        return False
              
    def detect_sift(self, img):

        # binarize for mask if image is not already binary
        if img.mode != 'L':
            to_b = np.array(img.convert('L'))
            doxapy.Binarization.update_to_binary(doxapy.Binarization.Algorithms.ISAUVOLA , to_b)
        else:
            to_b = np.array(img)

        # detect 
        sift = cv2.SIFT_create() #sigma=3) #, enable_precise_upscale=True)
        kp = sift.detect(to_b, None)
        kp = np.unique(cv2.KeyPoint_convert(kp).astype('int'), axis=0)
        return kp, Image.fromarray(to_b)

    
    def __getitem__(self, index):
        img = self.dataset.get_image(index)
        labels = self.get_label(index)

        kps, mask = self.detect_sift(img)
        tensor_mask = transforms.functional.to_tensor(mask).round()

        img = img.convert('RGB') if img.mode == 'L' else img  # make sure crops are consistent with three channels
        
        # iterate and filter the bad ones
        crops = []
        for kp in kps:
            if tensor_mask[0, kp[1], kp[0]] != 0:
                continue

            masked_crop = transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1]))


            if not self.check_crop(masked_crop):
                continue

            crops.append(transforms.functional.to_tensor(self.crop(img, kp[0], kp[1])))

        # normalize crops
        crops = self.normalize(torch.stack(crops))

        # sample crops if we have too many of them
        if self.num_samples != -1 and len(crops) > self.num_samples:
            idx = torch.linspace(0, len(crops) -1, self.num_samples).int()
            crops = crops[idx]
        
        labels = list(repeat(labels, crops.shape[0]))
        return crops, labels



class EvalSIFTColorSampler(DatasetWrapper):

    def __init__(self, *args, patch_size, num_samples, page_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.page_transforms = page_transforms

        self.patch_size = patch_size
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def crop(self, img, x,y, crop_size=None):
        crop_size = crop_size or (self.patch_size, self.patch_size)
        return img.crop((int(x-crop_size[0]/2), int(y-crop_size[1]/2), int(x+crop_size[0]/2), int(y+crop_size[1]/2)))

    def check_crop(self, tensor_crop):
        if tensor_crop.shape[1:] != (self.patch_size, self.patch_size):
            return False
        
        c = transforms.functional.rgb_to_grayscale(tensor_crop)
        if torch.sum(c < 0.005) < len(c.ravel()) * 0.5:
            if torch.sum(c > 0.995) < len(c.ravel()) * 0.95:
                    return True
      
        return False
              
    def detect_sift(self, img):

        # binarize for mask if image is not already binary
        if img.mode != 'L':
            to_b = np.array(img.convert('L'))
            doxapy.Binarization.update_to_binary(doxapy.Binarization.Algorithms.ISAUVOLA , to_b)
        else:
            to_b = np.array(img)

        # detect 
        sift = cv2.SIFT_create() #, enable_precise_upscale=True)
        kp = sift.detect(to_b, None)
        kp = np.unique(cv2.KeyPoint_convert(kp).astype('int'), axis=0)
        return kp, Image.fromarray(to_b)

    
    def __getitem__(self, index):
        img = self.dataset.get_image(index)
        labels = self.get_label(index)

        img = self.page_transforms(img) if self.page_transforms else img
        
        kps, mask = self.detect_sift(img)
        tensor_mask = transforms.functional.to_tensor(mask).round()

        img = img.convert('RGB') if img.mode == 'L' else img  # make sure crops are consistent with three channels
        
        # iterate and filter the bad ones
        crops = []
        for kp in kps:
            if tensor_mask[0, kp[1], kp[0]] != 0:
                continue

            # masked_crop = transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1]))

            crop = transforms.functional.to_tensor(self.crop(img, kp[0], kp[1]))
            if not self.check_crop(crop):   
                continue

            crops.append(transforms.functional.to_tensor(self.crop(img, kp[0], kp[1])))

        # normalize crops
        crops = self.normalize(torch.stack(crops)) # self.normalize(torch.stack(crops))

        # sample crops if we have too many of them
        if self.num_samples != -1 and len(crops) > self.num_samples:
            idx = torch.linspace(0, len(crops) -1, self.num_samples).int()
            crops = crops[idx]
        
        labels = list(repeat(labels, crops.shape[0]))
        return crops, labels


class SamplerWithBinary(DatasetWrapper):

    def __init__(self, *args, page_transforms, patch_size, num_keypoints, num_samples, mask_type='binary', binarization='SAUVOLA', **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.mask_type = mask_type
        self.binarization = binarization

        self.patch_size = patch_size
        self.page_transforms = page_transforms
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def crop(self, img, x,y, crop_size=None):
        crop_size = crop_size or (self.patch_size, self.patch_size)
        return img.crop((int(x-crop_size[0]/2), int(y-crop_size[1]/2), int(x+crop_size[0]/2), int(y+crop_size[1]/2)))

    def check_crop(self, tensor_crop):
        if tensor_crop.shape[1:] != (self.patch_size, self.patch_size):
            return False
        
        if torch.sum(1-tensor_crop.round()) < 0.08*len(tensor_crop.ravel()):
            return False
        if torch.sum(1-tensor_crop.round()) > 0.8*len(tensor_crop.ravel()):
            return False
        return True
              
    def detect_sift(self, img):

        # binarize for mask if image is not already binary
        if img.mode != 'L':
            to_b = np.array(img.convert('L'))
            doxapy.Binarization.update_to_binary(getattr(doxapy.Binarization.Algorithms, self.binarization, 'ISAUVOLA') , to_b)
        else:
            to_b = np.array(img)

        # detect 
        sift = cv2.SIFT_create()
        kp = list(sift.detect(to_b, None))
        kp = np.unique(np.array(cv2.KeyPoint_convert(kp)).astype('int'), axis=0)
        np.random.shuffle(kp)
        return kp, Image.fromarray(to_b)
        
    def __getitem__(self, index):
        image = self.dataset.get_image(index)
        labels = self.get_label(index)

        img = self.page_transforms(image)
        kps, mask = self.detect_sift(img)

        img = img.convert('RGB') if img.mode == 'L' else img  # make sure crops are consistent with three channels
        
        # iterate and filter the bad ones
        crops, targets = [], []
        for kp in kps:
            masked_crop = transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1]))

            if not self.check_crop(masked_crop):
                continue

            # input
            crop = transforms.functional.to_tensor(self.crop(img, kp[0], kp[1]))
            crops.append(crop)

            # output
            if self.mask_type == 'rgb':
                targets.append(crop)
            elif self.mask_type == 'binary':
                targets.append(transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1])))
            elif self.mask_type == 'gray':
                targets.append(transforms.functional.rgb_to_grayscale(transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1]))))
            else:
                raise ValueError('Unknown mask type: {}'.format(self.mask_type))

        if not crops:
            return None
        
        # normalize crops
        crops = torch.stack(crops)
        targets = torch.stack(targets)

        # sample crops if we have too many of them
        if len(crops) > self.num_samples:
            idx = torch.linspace(0, len(crops) -1, self.num_samples).int()
            crops = crops[idx]
            targets = targets[idx]


        if labels:
            return crops, targets, labels
        else:
            return crops, targets
    

class TrainSIFTSampler(DatasetWrapper):

    def __init__(self, *args, patch_size, num_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

        self.patch_size = patch_size
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def crop(self, img, x,y, crop_size=None):
        crop_size = crop_size or (self.patch_size, self.patch_size)
        return img.crop((int(x-crop_size[0]/2), int(y-crop_size[1]/2), int(x+crop_size[0]/2), int(y+crop_size[1]/2)))

    def check_crop(self, tensor_crop):
        if tensor_crop.shape[1:] != (self.patch_size, self.patch_size):
            return False
        
        c = tensor_crop
        # white > 95 %
        if torch.sum(c == 1) < c.ravel().shape[0] * 0.99: #0.95
            return True
    
        return False
              
    def detect_sift(self, img):

        # binarize for mask if image is not already binary
        if img.mode != 'L':
            to_b = np.array(img.convert('L'))
            doxapy.Binarization.update_to_binary(doxapy.Binarization.Algorithms.ISAUVOLA , to_b)
        else:
            to_b = np.array(img)

        # detect 
        sift = cv2.SIFT_create() #sigma=3) #, enable_precise_upscale=True)
        kp = sift.detect(to_b, None)
        kp = np.unique(cv2.KeyPoint_convert(kp).astype('int'), axis=0)
        return kp, Image.fromarray(to_b)

    
    def __getitem__(self, index):
        img = self.dataset.get_image(index)
        labels = self.get_label(index)

        kps, mask = self.detect_sift(img)
        tensor_mask = transforms.functional.to_tensor(mask).round()

        img = img.convert('RGB') if img.mode == 'L' else img  # make sure crops are consistent with three channels
        
        # iterate and filter the bad ones
        crops = []
        for kp in kps:
            if tensor_mask[0, kp[1], kp[0]] != 0:
                continue

            masked_crop = transforms.functional.to_tensor(self.crop(mask, kp[0], kp[1]))


            if not self.check_crop(masked_crop):
                continue

            crops.append(transforms.functional.to_tensor(self.crop(img, kp[0], kp[1])))

        # normalize crops
        crops = self.normalize(torch.stack(crops))

        # sample crops if we have too many of them
        random.shuffle(crops)
        if self.num_samples != -1 and len(crops) > self.num_samples:
            idx = torch.linspace(0, len(crops) -1, self.num_samples).int()
            crops = crops[idx]
        
        labels = list(repeat(labels, crops.shape[0]))
        return crops, labels
