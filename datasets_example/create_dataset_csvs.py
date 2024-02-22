
import glob, re, os
from pathlib import Path

from tqdm import tqdm
import pandas as pd

EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

class CSVWriter:

    def __call__(self):
        self.get_csv()

    def get_images_and_labels(self):
        imgs, labels = [], {}
        files = sorted(list(glob.glob(f'{self.root}/**/*', recursive=True)))

        print(f'Found {len(files)} files.')
        for file in tqdm(files):
            if not any([file.endswith(ext) for ext in EXTENSIONS]):
                continue
            f = Path(file).name
            p = getattr(self, 'label_prefix', '')
            if self.labels:
                for name, regex in self.labels.items():
                    n = 'raw_' + name 
                    r = '_'.join(re.search(regex, f).groups())
                    labels[n] = labels.get(n, [])
                    r = p + "_" + r if p else r
                    labels[n].append(r)
            imgs.append(file)
        return imgs, labels

    def get_csv(self):
        imgs, labels = self.get_images_and_labels()
        
        modes = [self.mode] * len(imgs)
        names = [self.name] * len(imgs)
        label2int = {}
        if labels:
            for label in labels.keys():
                ids = set(labels[label])
                label_to_int = {writer: i for i, writer in enumerate(ids)}
                label2int[f"{label.split('_')[1]}"] = [label_to_int[l] for l in labels[label]]
                    
            labels.update(label2int)

        df_dict = {
            'dataset' : names,
            'imgs' : imgs,
            'mode' : modes,
            **labels
        }
        df = pd.DataFrame(df_dict)
        df.to_csv(self.out)

class ICDAR2017_SelfSupervised_Validation(CSVWriter):
    name = 'icdar2017-val'
    labels = {
        'writer' : '(\d+)',
        'page' : '\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/icdar2017-train_binarized_split/val'
    mode = 'bw'
    out = 'test.csv'

class ICDAR2017_Train_Patches(CSVWriter):
    name = 'icdar2017-train-binarized-patches_clustered'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\d+)',
        'page' : '\d+_\d+-\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/resources/icdar2017_train_sift_patches_binarized'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'


class ICDAR2017_Train_Patches_Color(CSVWriter):
    name = 'icdar2017-train-color-patches_clustered'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\d+)',
        'page' : '\d+_\d+-\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/icdar2017-training_patches5k_clustered'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class ICDAR2017_Train_Patches10k(CSVWriter):
    name = 'icdar2017-train-binarized-patches_clustered10k'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\d+)',
        'page' : '\d+_\d+-\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/icdar2017_10kclustered_centered'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class ICDAR2017_Train_Patches5k_NoFilter(CSVWriter):
    name = 'icdar2017-train-binarized-patches_clustered5k_nofilter'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\d+)',
        'page' : '\d+_\d+-\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/resources/icdar2017-train-binarized-patches-5kclusters_nofilter'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class ICDAR2017_Test(CSVWriter):
    name = 'icdar2017-test-binarized'
    labels = {
        'writer' : '(\d+)',
        'page' : '\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/icdar2017-test_binarized'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class ICDAR2017_Test_Color(CSVWriter):
    name = 'icdar2017-test-color'
    labels = {
        'writer' : '(\d+)',
        'page' : '\d+-IMG_MAX_(\d+)'
    }
    root = '/data/mpeer/icdar2017-test-color'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class HisFrag20_Train(CSVWriter):
    name = 'hisfrag20_train'
    labels = {
        'writer' : '(\d+)',
        'page' : '\d+_(\d+)',
        'fragment' : '\d+_\d+_(\d+)'
    }
    root = '/data/mpeer/hisfrag20'
    mode = 'color'
    out = 'datasets/' + name + '.csv'

class HisFrag20_Test(CSVWriter):
    name = 'hisfrag20_test'
    labels = {
        'writer' : '(\d+)',
        'page' : '\d+_(\d+)',
        'fragment' : '\d+_\d+_(\d+)'
    }
    root = '/data/mpeer/hisfrag20_test'
    mode = 'color'
    out = 'datasets/' + name + '.csv'


class HisFrag20_Train_Cluster(CSVWriter):
    name = 'hisfrag20_train_patches_clustered'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\d+)',
        'page' : '\d+_\d+_(\d+)',
        'fragment' : '\d+_\d+_\d+_(\d+)'
    }
    root = '/data/mpeer/resources/hisfrag20_train_clusters'
    mode = 'color'
    out = 'datasets/' + name + '.csv'



class GRK50_BinarizedPatches(CSVWriter):
    name = 'GRK50_patches_binarized_clustered_5000'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\w+)_\d+_\d+',
        'page' : '\d+_\w+_(\d+)_\d+',
    }
    root = '/data/mpeer/grk50_binarized_scaled_0p5_clustered5k'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class GRK50_BinarizedPatchesTest(CSVWriter):
    name = 'GRK50_patches_test'
    labels = {
        'writer' : '(\w+)_\d+_\d+',
        'page' : '\w+_(\d+)_\d+',
    }
    root = '/data/mpeer/grk50_binarized_scaled_0p5_test_patches'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'

class GRK50_ColorPatches(CSVWriter):
    name = 'GRK50_patches_color_clustered2000_rsift'
    labels = {
        'cluster' : '(\d+)',
        'writer' : '\d+_(\w+)_\d+_\d+',
        'page' : '\d+_\w+_(\d+)_\d+',
    }
    root = '/data/mpeer/grk50_scaled_0p5_RSIFT2000C_10kkp'
    mode = 'color'
    out = 'datasets/grk50/' + name + '.csv'


class GRK50_ColorPatchesTest(CSVWriter):
    name = 'GRK50_patches_color_rsift_test'
    labels = {
        'writer' : '(\w+)_\d+_\d+',
        'page' : '\w+_(\d+)_\d+',
    }
    root = '/data/mpeer/grk50_scaled_0p5_RSIFT_10kkp_test'
    mode = 'color'
    out = 'datasets/grk50/' + name + '.csv'


class ICDAR2013_Train(CSVWriter):
    name = 'icdar2013_train_patches_1000'
    labels = {
        'cluster' : '(\d+)_\d+_\d+_\d+',
        'writer' : '\d+_(\d+)_\d+_\d+',
        'page' : '\d+_\d+_(\d+)_\d+',
    }
    root = '/data/mpeer/icdar2013_train_1000'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'


class ICDAR2013_Test(CSVWriter):
    name = 'icdar2013_test'
    labels = {
        'writer' : '(\d+)_\d+',
        'page' : '\d+_(\d+)',
    }
    root = '/data/mpeer/icdar2013_test'
    mode = 'bw'
    out = 'datasets/' + name + '.csv'


class Papyrow_Train(CSVWriter):
    name = 'papyrow_patches_train'
    labels = {
        'cluster' : '(\d+)_\w+_\d{1,2}[_-]\d{1,3}_\d+',
        'writer' : '\d+_(\w+)_\d{1,2}[_-]\d{1,3}_\d+',
        'page' : '\d+_\w+_(\d{1,2}[_-]\d{1,3})_\d+',
    }
    root = '/data/mpeer/papyrow_clustered_5000_scaled0p5'
    mode = 'bw'
    out = 'datasets/grk50/' + name + '.csv'

class Papyrow_Test(CSVWriter):
    name = 'papyrow_patches_test'
    labels = {
        'writer' : '(\w+)_\d{1,2}[_-]\d{1,3}_\d+',
        'page' : '\w+_(\d{1,2}[_-]\d{1,3})_\d+',
    }
    root = '/data/mpeer/papyrow_scaled0p5_test'
    mode = 'bw'
    out = 'datasets/grk50/' + name + '.csv'



if __name__ == '__main__':
    ICDAR2017_Test_Color()()
