# 3D-Medical-Imaging-Preprocessing-All-you-need
This Repo Will contain the Preprocessing Code for 3D Medical Imaging

From the last year of my undergrade studies I was very queries about Biomedical Imaging. But until the starting my master I don't have the chance to go deep into the medical imaging. Like most people at the begining I also suffered and was bit confussed about few thing. In this node book I will try to easy explain commonly used Preprocessing in Medical Imaging.

In this tutorial we will be using Public Abdomen Dataset From: Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge Link: https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

Reference: https://github.com/DLTK/DLTK

In this Notebook we will cover
1. **Reading Nifti Data and ploting**
2. **Different Intensity Normalization Approaches**
3. **Resampling 3D CT data**
4. **Cropping and Padding CT data**
5. **Histogram equalization**
6. **Maximum Intensity Projection (MIP)**


`if You want know about MRI Histogram Matching, Histogram Equalization and Registration, You could have a look to my repo`

* https://github.com/fitushar/Brain-Tissue-Segmentation-Using-Deep-Learning-Pipeline-NeuroNet
* https://github.com/fitushar/Registration-as-Data-Augumentation-for-CT--DATA


`To Learn about Segmentation`

* **Brain Tissue Segmentation**, 3D : https://github.com/fitushar/Brain-Tissue-Segmentation-Using-Deep-Learning-Pipeline-NeuroNet 
* **Chest-Abdomen-Pelvis (Segmentation)** 3D DenseVnet :https://github.com/fitushar/DenseVNet3D_Chest_Abdomen_Pelvis_Segmentation_tf2

* **3D-Unet** : https://github.com/fitushar/3DUnet_tensorflow2.0


## Libraries need
```ruby
* SimpleITK
* numpy
* scipy
* skimage
* cv2
```  
### **Reading Nifti Data and ploting**
```ruby
ct_path='D:/Science/Github/3D-Medical-Imaging-Preprocessing-All-you-need/Data/img0001.nii.gz'
ct_label_path='D:/Science/Github/3D-Medical-Imaging-Preprocessing-All-you-need/Data/label0001.nii.gz'

# CT
img_sitk  = sitk.ReadImage(ct_path, sitk.sitkFloat32) # Reading CT
image     = sitk.GetArrayFromImage(img_sitk) #Converting sitk_metadata to image Array
# Mask
mask_sitk = sitk.ReadImage(ct_label_path,sitk.sitkInt32) # Reading CT
mask      = sitk.GetArrayFromImage(mask_sitk)#Converting sitk_metadata to image Array
```  
![ct_mask](https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need/blob/master/figure/CT.PNG)

### Intensity Normalization
```ruby
def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    np_img = np.clip(np_img, -1000., 800.).astype(np.float32)
    return np_img


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret
``` 
![Normalization](https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need/blob/master/figure/IntensityNormalization.PNG)
