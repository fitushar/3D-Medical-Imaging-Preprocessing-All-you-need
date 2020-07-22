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
## **Reading Nifti Data and ploting**
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

## Intensity Normalization
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

## Resampling 
```ruby
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


```ruby
![Resampling](https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need/blob/master/figure/Resampled.PNG)


## Crop or Padding 
```ruby
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


```ruby
![Cropandpad](https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need/blob/master/figure/Crop_or_padding.PNG)
