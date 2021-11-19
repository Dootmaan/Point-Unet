import torch
import torch.utils.data
import SimpleITK as sitk
import os
from scipy import ndimage
import numpy as np
import cv2

class ResUNetDatasetPatch3D(torch.utils.data.Dataset):
    def __init__(self, path, train=True,size=128.0,patch_size=64):
        self.size=float(size)
        self.data=list()
        self.label_seg=list()
        self.patch_size=patch_size

        if train == True:
            for i,folder in enumerate(os.listdir(path)):
                
                #88Z has no lesion mask(healthy),
                if folder=='88Z':
                    continue

                if os.path.exists(path+'/'+folder+'/'+'lesion.nrrd'):
                    if i<0.8*len(os.listdir(path)):
                        print('Processing training set:',folder)
                        image=sitk.ReadImage(path+'/'+folder+'/'+'image.nrrd')
                        image_arr=sitk.GetArrayFromImage(image)
                        lung=sitk.ReadImage(path+'/'+folder+'/'+'lung.nrrd')
                        lung_arr=sitk.GetArrayFromImage(lung)
                        lesion=sitk.ReadImage(path+'/'+folder+'/'+'lesion.nrrd')
                        lesion_arr=sitk.GetArrayFromImage(lesion)
                        self.truncate_hu(image_arr)
                        image_arr=self.normalization(image_arr)
                        cropImg,cropMask=self.cropCT(image_arr*lung_arr,lesion_arr*lung_arr)
                        
                        if cropImg.shape[0]==0.5*size and cropImg.shape[1]==size and cropImg.shape[2]==size:

                            for j in range(0,int(0.5*size),0.5*patch_size):   # So right now 1 case = 8 patches
                                for k in range(0,int(size),patch_size):
                                    for l in range(0,int(size),patch_size):
                                        self.data.append(cropImg[j:(j+0.5*patch_size),k:(k+patch_size),l:(l+patch_size)])
                                        self.label_seg.append(cropMask[j:(j+0.5*patch_size),k:(k+patch_size),l:(l+patch_size)])
                                        
        else:
            for i,folder in enumerate(os.listdir(path)):
                if os.path.exists(path+'/'+folder+'/'+'lesion.nrrd'):
                    if i>=0.8*len(os.listdir(path)):
                        print('Processing testing set:',folder)
                        image=sitk.ReadImage(path+'/'+folder+'/'+'image.nrrd')
                        image_arr=sitk.GetArrayFromImage(image)
                        lung=sitk.ReadImage(path+'/'+folder+'/'+'lung.nrrd')
                        lung_arr=sitk.GetArrayFromImage(lung)
                        lesion=sitk.ReadImage(path+'/'+folder+'/'+'lesion.nrrd')
                        lesion_arr=sitk.GetArrayFromImage(lesion)
                        self.truncate_hu(image_arr)
                        image_arr=self.normalization(image_arr)
                        cropImg,cropMask=self.cropCT(image_arr*lung_arr,lesion_arr*lung_arr)

                        if cropImg.shape[0]==0.5*size and cropImg.shape[1]==size and cropImg.shape[2]==size:

                            # For testing phase, use original label without patching.
                            self.data.append(cropImg)
                            self.label_seg.append(cropMask)


    def truncate_hu(self,image_array):
        image_array[image_array > 500] = 500
        image_array[image_array <-1000] = -1000
    
    def normalization(self,image_array):
        max = image_array.max()
        min = image_array.min()
        #归一化
        image_array = 1.0*(image_array - min)/(max - min)
        #image_array = image_array.astype(int)#整型
        return image_array
    
    def cropCT(self,img,mask):
        result=np.where(img!=0)
        z_list=result[0]
        x_list=result[1]
        y_list=result[2]

        x_max=x_list.max()
        x_min=x_list.min()
        y_max=y_list.max()
        y_min=y_list.min()
        z_max=z_list.max()
        z_min=z_list.min()

        return self.normalization(ndimage.interpolation.zoom(img[z_min:z_max,x_min:x_max,y_min:y_max],[0.5*self.size/(z_max-z_min),self.size/(x_max-x_min),self.size/(y_max-y_min)],order=3)),self.normalization(ndimage.interpolation.zoom(mask[z_min:z_max,x_min:x_max,y_min:y_max],[0.5*self.size/(z_max-z_min),self.size/(x_max-x_min),self.size/(y_max-y_min)],order=3))

    def __len__(self):
        return len(self.label_seg)

    def __getitem__(self, index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None

        return (self.data[index],self.label_seg[index])
    
