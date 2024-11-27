import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from initialpreprocess import get_all_nodules

def view_slice(image, slice_num, name):
    slice_2d = sitk.GetArrayViewFromImage(image)[slice_num, :, :]
    plt.imshow(slice_2d, cmap='gray')
    plt.axis('off')
    plt.savefig(name)


# SHOW A SLICE OF A 3D IMAGE FROM A DICOM SERIES
"""
image_folder = "../../dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI/LIDC-IDRI-0079/01-01-2000-37490/3278.000000-05159"
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(image_folder)
reader.SetFileNames(dicom_names)
image_3d = reader.Execute()
"""

# SHOW A SLICE OF A 3D IMAGE FROM A NIFTI FILE

#test_path = "nodule_LIDC-IDRI-0079_26.nii"
#test_path = "nodules/nodule_LIDC-IDRI-0874_nodule1.nii"
#image_3d = sitk.ReadImage(test_path)

nodules = get_all_nodules()
nodule_tup = nodules[0]


for index, nodule_tup in enumerate(nodules):
    nodule = nodule_tup[0]
    malignancy = nodule_tup[1]
    malignancy_truth = nodule_tup[2]
    diameter = nodule_tup[3]
    centroid = nodule_tup[4]

    image_3d =  sitk.GetImageFromArray(nodule)
    print(image_3d.GetSize())
    print(image_3d.GetSpacing())

    for slice_num in range(image_3d.GetSize()[2]):    
        view_slice(image_3d, slice_num, f"noduleimages/nodule_{index}_slice_{slice_num}")
