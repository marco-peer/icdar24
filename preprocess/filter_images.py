import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from joblib import Parallel, delayed

def get_kps(imgp):
    img = cv2.imread(str(imgp), 0)
    img = cv2.Canny(img, 100, 200)
    sift = cv2.SIFT_create()
    kp = sift.detect(img,None)
    return kp

imgs = [p for p in Path('/data/mpeer/data_masks/').rglob('*') if not os.path.isdir(p) and p.suffix in ['.png', '.jpg', '.tif']]

invalid_imgs = []

def filter_imgs(imgp):
    kps = get_kps(imgp)
    if len(kps) < 1000:
        return imgp

results = Parallel(n_jobs=8, verbose=9)(delayed(filter_imgs)(img) for img in imgs)


with open("invalid_imgs.txt", 'w') as f:
    for res in results:
        if res:
            f.write(str(res) + '\n')

