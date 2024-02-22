
import os, shutil
from pathlib import Path

# opening the file in read mode 
my_file = open("invalid_imgs.txt", "r") 
  
# reading the file 
data = my_file.read().split("\n")


tar = '/data/mpeer/data_masks/invalid'

if not os.path.exists(tar):
    os.mkdir(tar)

# print(os.path.join(tar, Path(data[0]).name))

def move_to_tar(imgp, tar):
    shutil.move(imgp, os.path.join(tar, Path(imgp).name))

# for d in data:
#     move_to_tar(d, tar)

move_to_tar('invalid_imgs.txt', tar)