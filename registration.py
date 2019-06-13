
import nibabel as nib
import skimage.io as io
import numpy as np
import os
import re
import time
import copy
path = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge"
mammalpath = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0gt"
c0title = "_C0.nii.gz"
lgetitle = "_LGE.nii.gz"
t2title = "_T2.nii.gz"

def split_manual(filename, mammalpath):
    img=nib.load(filename)
    img_r=img.get_fdata()
    img_r=np.squeeze(img_r)

    img_1=nib.load(filename)
    img_c=img_1.get_fdata()
    img_c=np.squeeze(img_c)

    img_2=nib.load(filename)
    img_v=img_2.get_fdata()
    img_v=np.squeeze(img_v)
    hdr = img.header
    [rows, cols, valume] = img_r.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(valume):
                if img_r[i, j, k] == 200 :
                    
                    img_r[i, j, k] = 1
                    img_c[i, j, k] = 0
                    img_v[i, j, k] = 0 
                if img_r[i, j, k ] == 500 :
                    img_r[i, j, k] = 0
                    img_c[i, j, k] = 1
                    img_v[i, j, k] = 0
                if img_r[i, j, k ] == 600 :
                    img_r[i, j, k] = 0
                    img_c[i, j, k] = 0
                    img_v[i, j, k] = 1
    r_mannual = re.sub('_C0_manual.nii.gz', "_C0_r_manual.nii.gz", filename)
    c_mannual = re.sub('_C0_manual.nii.gz', "_C0_c_manual.nii.gz", filename)
    v_mannual = re.sub('_C0_manual.nii.gz', "_C0_v_manual.nii.gz", filename)
    
    
    save_img_r = nib.Nifti1Image(img_r, img.affine, img.header)
    nib.save(save_img_r, r_mannual)
    save_img_c = nib.Nifti1Image(img_c, img_1.affine, img_1.header)
    nib.save(save_img_c, c_mannual)
    save_img_v = nib.Nifti1Image(img_v, img_2.affine, img_2.header)
    nib.save(save_img_v, v_mannual)
    return r_mannual, c_mannual, v_mannual

def merge_rcv_mannual(r_filename, c_filename, v_filename):
    
    r_img=nib.load(r_filename)
    r_img_arr=r_img.get_fdata()
    r_img_arr=np.squeeze(r_img_arr)
    c_img=nib.load(c_filename)
    c_img_arr=c_img.get_fdata()
    c_img_arr=np.squeeze(c_img_arr)
    v_img=nib.load(v_filename)
    v_img_arr=v_img.get_fdata()
    v_img_arr=np.squeeze(v_img_arr)
    hdr = r_img.header
    [rows, cols, valume] = r_img_arr.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(valume):
                if r_img_arr[i, j, k] >0.4:
                    r_img_arr[i, j, k] = 1
                if c_img_arr[i, j, k] > 0.4:
                    r_img_arr[i, j, k] = 2
                if v_img_arr[i, j, k] >0.4:
                    r_img_arr[i, j, k] = 3
    another_img = nib.Nifti1Image(r_img_arr, r_img.affine, hdr)
    nib.save(another_img, \
        re.sub('_C02LGE_r_manual.nii.gz', "_C02LGE_rcv_manual.nii.gz", r_filename))

def mammaltransfer(r_mannual, c_mannual, v_mannual):
    tmp = re.sub('_C0_r_manual.nii.gz', "_LGE.nii.gz", r_mannual)
    lgename = re.sub('c0gt', "c0t2lge", tmp)
    C02LGE_r_manual = re.sub('_C0_r_manual.nii.gz', "_C02LGE_r_manual.nii.gz", r_mannual)
    matname = re.sub('_LGE.nii.gz', "_C02LGE.mat", lgename)
    cmd = "flirt -in " + r_mannual + \
        " -ref " + lgename + \
        " -out " + C02LGE_r_manual + \
        " -init " + matname + " -applyxfm"
    # print(cmd)
    os.system(cmd)
    tmp = re.sub('_C0_c_manual.nii.gz', "_LGE.nii.gz", c_mannual)
    lgename = re.sub('c0gt', "c0t2lge", tmp)
    C02LGE_c_manual = re.sub('_C0_c_manual.nii.gz', "_C02LGE_c_manual.nii.gz", c_mannual)
    matname = re.sub('_LGE.nii.gz', "_C02LGE.mat", lgename)
    cmd = "flirt -in " + c_mannual + \
        " -ref " + lgename + \
        " -out " + C02LGE_c_manual + \
        " -init " + matname + " -applyxfm"
    # print(cmd)
    os.system(cmd)
    tmp = re.sub('_C0_v_manual.nii.gz', "_LGE.nii.gz", v_mannual)
    lgename = re.sub('c0gt', "c0t2lge", tmp)
    C02LGE_v_manual = re.sub('_C0_v_manual.nii.gz', "_C02LGE_v_manual.nii.gz", v_mannual)
    matname = re.sub('_LGE.nii.gz', "_C02LGE.mat", lgename)
    cmd = "flirt -in " + v_mannual + \
        " -ref " + lgename + \
        " -out " + C02LGE_v_manual + \
        " -init " + matname + " -applyxfm"
    # print(cmd)
    os.system(cmd) 
    return C02LGE_r_manual, C02LGE_c_manual, C02LGE_v_manual

def regestration(path):
    for _, _, filelist in os.walk(path):
        for i in filelist:
            if (None != re.search(c0title, i)):
                filename = path  + "/" + i
                os.system("flirt -in " + filename + \
                    " -ref " + re.sub('_C0.nii.gz', "_LGE.nii.gz", filename) + \
                    " -out " + re.sub('_C0.nii.gz', "_C02LGE.nii.gz", filename) + \
                    " -omat "+ re.sub('_C0.nii.gz', "_C02LGE.mat", filename) + \
                    " -dof 12")
            elif (None != re.search(t2title, i)):
                pass
            elif (None != re.search(lgetitle, i)):
                pass

def rcvhandler(mammalpath):
    for _, _, filelist in os.walk(mammalpath):
        for i in filelist:
            if (None != re.search("_C0_manual.nii.gz", i)):
                filename = mammalpath  + "/" + i
                r_mannual, c_mannual, v_mannual =split_manual(filename, mammalpath)
                c02lge_r_mannual, c02lge_c_mannual, c02lge_v_mannual = mammaltransfer(r_mannual, c_mannual,v_mannual) 
                merge_rcv_mannual(c02lge_r_mannual, c02lge_c_mannual, c02lge_v_mannual)

def main():
    # regestration(path) 
    # 
    
    # manual
    rcvhandler(mammalpath)
    
    

if __name__ == "__main__":
    main()