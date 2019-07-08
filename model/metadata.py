import SimpleITK as sitk
import os
import re
import time
or_path = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge"
dst_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/final/v2"
import nibabel as nib
def read_img(path):
    img = sitk.ReadImage(path)
    return img

def get_or_path():
    path_dict = {}
    for root, dirs, files in os.walk(or_path):
            for i in files:
                if (None != re.search("_LGE.nii.gz", i)):
                    tmp = i.replace("_LGE.nii.gz","")
                    idx = tmp.replace("patient","")
                    new_path = os.path.join(or_path, i)
                    path_dict[int(idx)]= new_path
    return path_dict

def get_dst_path():
    path_dict = {}
    for root, dirs, files in os.walk(dst_path):
            for i in files:
                if (None != re.search(".nii.gz", i)):
                    tmp = i.replace(".nii.gz","")
                    idx = tmp
                    print("idx:",idx)
                    new_path = os.path.join(dst_path, i)
                    path_dict[int(idx)]= new_path
    return path_dict
                   
or_path_dict = get_or_path()  
dst_path_dict = get_dst_path()
for i in range(6,46):
        or_img_path = or_path_dict[i]
        or_img = read_img(or_img_path)
        dst_img_path = dst_path_dict[i]
        dst_img = read_img(dst_img_path)        
        dst_img.CopyInformation(or_img)
        print(dst_img)
        sitk.WriteImage(dst_img, dst_img_path)
