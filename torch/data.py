import filecmp
import shutil
import glob
import os

hold_name = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True))
for i in range(len(hold_name)):
    hold_name[i] = os.path.basename(hold_name[i])

origin_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/66/*.png', recursive=True))
origin_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/66/*.png', recursive=True))
origin_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation/66/*.png', recursive=True))

custom_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/train/66/*.png', recursive=True))
custom_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/test/66/*.png', recursive=True))
custom_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/66/*.png', recursive=True))

for i in range(len(custom_train_path)):
    custom_train_path[i] = os.path.basename(custom_train_path[i]) #97
for i in range(len(origin_train_path)):
    origin_train_path[i] = os.path.basename(origin_train_path[i]) #80 

diff = list(set(origin_train_path) - set(custom_train_path))

for check in diff:
    print(check)
        