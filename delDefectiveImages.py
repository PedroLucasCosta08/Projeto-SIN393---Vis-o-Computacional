import os
from PIL import Image
import numpy as np
import shutil

data_dir= r'/home/pedrocosta/Dev/Datasets/CropPestAndDiseaseDetection/CropPestAndDiseaseDetection'
bad_img_list=[]
total=0
good=0
bad=0

classes=sorted(os.listdir(data_dir))
for klass in classes:
    good_class=0
    bad_class=0
    total_class=0
    msg=f'processing class {klass}'
    print(msg, '\r', end= '')
    classpath=os.path.join(data_dir, klass)
    flist=sorted(os.listdir(classpath))
    for f in flist:
        total +=1
        total_class +=1
        fpath=os.path.join(classpath,f)
        try:
            img= Image.open(fpath) 
            array=np.asarray(img)
            good +=1
            good_class +=1
        except:
            bad_img_list.append(fpath)
            bad +=1
            bad_class +=1
    
    msg=f'class {klass} contains {total_class} files, {good_class} are valid image files and {bad_class} defective image files'
    print (msg)

msg=f'the dataset contains {total} image files, {good} are valid image files and {bad} are defective image files'
print (msg)
if bad>0:
    ans=input('to print a list of defective image files enter P, to not print press Enter')
    if ans == 'P' or ans == 'p':
        for f in bad_img_list:
            print (f)
    
    delete=input('Do you wish to delete these defective imgs ? [y/n] ')
    if delete == 'Y' or delete == 'y':
        print('Img files deleted: ')
        for img in bad_img_list:
            if os.path.exists(img):
                print('Deleting the defective imgs',  '\r', end='')
                os.remove(img)
                print(f'{img} deleted!')
            else: 
                print('Couldnt find ', img, '\n')
