import logging

import numpy as np
import pandas as pd
from PIL import Image

from dataloading.wrapper import WrapableDataset

label_names = ['writer', 'page', 'cluster', 'line']

def label2int(labels):
    unique_labels = list(set(labels))
    label2int_dict = {l : unique_labels.index(l) for l in unique_labels}
    int2label_dict = {unique_labels.index(l) : l for l in unique_labels}

    int_labels = [label2int_dict[l] for l in labels]
    return int_labels, labels, label2int_dict, int2label_dict

class DocumentDataset(WrapableDataset):

    def __init__(self, df_path):
        self.df_path = df_path
        self.df = pd.read_csv(df_path, index_col=0)
        self.imgs = self.df['imgs'].tolist()

        ### labels
        self.label_names = []
        self.labels = {}
        self.raw_labels = {}
        self.label2int = {}
        self.int2label = {}
        
        for label_name in label_names:
            if label_name not in self.df.columns.tolist():
                continue
            self.label_names.append(label_name)
            int_labels, _, label2int_dict, int2label_dict = label2int(self.df[label_name].tolist())
            self.labels[label_name] = int_labels
            self.label2int[label_name] = label2int_dict
            self.int2label[label_name] = int2label_dict

        self.packed_labels = np.stack([self.labels[l] for l in self.label_names], axis=1) if self.label_names else []
        self.df = None
        self.loader = lambda x: Image.open(x).convert('RGB')
        logging.info(f'Loaded {len(self)} images from {df_path}')
              
    def get_image(self, index):
        img = self.imgs[index]
        img = self.loader(img)
        return img 

    def get_label(self, index):
        # if no labels are available
        if not self.label_names:
            return 
        
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __len__(self):
        return len(self.imgs)


