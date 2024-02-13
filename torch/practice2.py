import sys, os
import cv2
import glob
import pandas as pd
import numpy as np
import random
import argparse
import math
import shutil
 
def count_files_in_directory(directory_path):
    try:
        # 디렉토리 내 파일의 개수 세기
        file_count = len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])
        return file_count
    except FileNotFoundError:
        os.makedirs(directory_path, exist_ok=True)
        return 0
    except Exception as e:
        # 기타 예외 처리
        return None
   
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference flowmap using onnx model')
    parser.add_argument('--outdir', type=str, default="./20240206_Holder_name", help="")  
    args = parser.parse_args()
 
    val_raio = 0.15
 
    indir_list = 'C:/Users/QR22002/Desktop/chaeyun/dataset/train'
 
    class_list = ['100', '44',	'45',	'46',   '65',	
              '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	
              '75',	'76',	'77',	'78',	'79',	'80',	'81',	
              '82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',	'90']
 
 
    for class_ in class_list:
       
        trainset_path = 'C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/' + class_
        validationset_path = 'C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation/' + class_
        testset_path = 'C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/' + class_
        os.makedirs(trainset_path, exist_ok=True)
        os.makedirs(validationset_path, exist_ok=True)
        os.makedirs(testset_path, exist_ok=True)
 
        list0 = glob.glob(indir_list + '/' + class_ + '/*.png')
        img_list = list0
        random.shuffle(img_list)
        check = [False for i in range(len(img_list))]
        curr_class_total_length = len(img_list)
 
        curr_class_val_num = math.floor(curr_class_total_length * val_raio)

        if class_ == '100':
            continue
            for idx in range(20000):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], testset_path + '/' + str(img_name))
            for idx in range(20000, 40000):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], validationset_path + '/' + str(img_name))
            for idx in range(40000, len(img_list)):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], trainset_path + '/' + str(img_name))
                
        elif class_ == '44' or class_== '46':
            continue
            for idx in range(200):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], testset_path + '/' + str(img_name))
            for idx in range(200, 400):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], validationset_path + '/' + str(img_name))
            for idx in range(400, len(img_list)):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], trainset_path + '/' + str(img_name))
        
        elif class_ == '45':
            for idx in range(9):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], testset_path + '/' + str(img_name))
            for idx in range(9, 19):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], validationset_path + '/' + str(img_name))
            for idx in range(19, len(img_list)):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], trainset_path + '/' + str(img_name))
        
        else:
            continue
            for idx in range(0, curr_class_val_num):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], validationset_path + '/' + str(img_name))
            for idx in range(curr_class_val_num, len(img_list)):
                img_name = os.path.basename(img_list[idx])
                shutil.copy(img_list[idx], trainset_path + '/' + str(img_name))