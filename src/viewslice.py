import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def view_slice(image, slice_num):
    slice_2d = sitk.GetArrayViewFromImage(image)[slice_num, :, :]
    plt.imshow(slice_2d, cmap='gray')
    plt.axis('off')
    plt.savefig('slice.png')


# SHOW A SLICE OF A 3D IMAGE FROM A DICOM SERIES
"""
image_folder = "../../dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192"
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(image_folder)
reader.SetFileNames(dicom_names)
image_3d = reader.Execute()
"""

# SHOW A SLICE OF A 3D IMAGE FROM A NIFTI FILE
"""
test_path = "../../dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI_resampled/LIDC-IDRI-0042_01-01-2000-73529_3000906.000000-08135_resampled.nii"
image_3d = sitk.ReadImage(test_path)
"""

print(image_3d.GetSize())
print(image_3d.GetSpacing())

view_slice(image_3d, 1)