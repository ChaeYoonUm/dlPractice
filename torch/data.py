import filecmp
import shutil
import glob
import os

hold_name = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True))
for i in range(len(hold_name)):
    hold_name[i] = os.path.basename(hold_name[i])

origin_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/71/*.png', recursive=True))
origin_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/71/*.png', recursive=True))
origin_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation/71/*.png', recursive=True))

custom_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/train/71/*.png', recursive=True))
custom_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/test/71/*.png', recursive=True))
custom_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/71/*.png', recursive=True))

for i in range(len(origin_val_path)):
    origin_val_path[i] = os.path.basename(origin_val_path[i]) #97
for i in range(len(custom_val_path)):
    custom_val_path[i] = os.path.basename(custom_val_path[i]) #80 

# for check in _origin_val_path:  
#     src = 'C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/71/'
#     dst = 'C:/Users/QR22002/Desktop/chaeyun/data/junk4/'
#     if check in origin_val_path:
#         shutil.move(src+check, dst+check)
# exit()
diff = list(set(origin_val_path) - set(custom_val_path))
for check in diff:
    print(check)
exit()
for check in diff:
    if check in custom_train_path:
        src = 'C:/Users/QR22002/Desktop/chaeyun/custom_dataset/train/71/'
        dst = 'C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/71/'
        shutil.move(src+check, dst+check)
        