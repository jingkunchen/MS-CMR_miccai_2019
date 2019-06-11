
import nibabel as nib
import skimage.io as io
import numpy as np
import os
path = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge"

def split(filename):
    img=nib.load(filename)
    img_r=img.get_fdata()
    img_r=np.squeeze(img_r)
    img_c=img.get_fdata()
    img_c=np.squeeze(img_c)
    img_v=img.get_fdata()
    img_v=np.squeeze(img_v)
    hdr = img.header
    [rows, cols, valume] = img_r.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(valume):
                if img_r[i, j, k ] == 200 :
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
    save_img_r = nib.Nifti1Image(img_r, img.affine, hdr)
    nib.save(save_img_r, os.path.join(path,'patient4_C02LGE_r_manual.nii.gz'))
    save_img_c = nib.Nifti1Image(img_c, img.affine, hdr)
    nib.save(save_img_c, os.path.join(path,'patient4_C02LGE_c_manual.nii.gz'))
    save_img_v = nib.Nifti1Image(img_v, img.affine, hdr)
    nib.save(save_img_v, os.path.join(path,'patient4_C02LGE_v_manual.nii.gz'))

def merge_rcv(r_filename, c_filename, v_filename):
    print(r_filename,c_filename,v_filename)
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
    nib.save(another_img, os.path.join(path,'patient4_C02LGE_rcv_manual.nii.gz'))
                    

def main():
    merge_rcv(os.path.join(path, 'patient4_C02LGE_r_manual.nii.gz'),os.path.join(path, 'patient4_C02LGE_c_manual.nii.gz'),os.path.join(path, 'patient4_C02LGE_v_manual.nii.gz'))
    split(os.path.join(path, 'patient4_C0_manual.nii.gz'))

if __name__ == "__main__":
    main()