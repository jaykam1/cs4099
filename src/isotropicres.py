import SimpleITK as sitk
import numpy as np
import os

# FUNCTION TAKES AN IMAGE AND RETURNS A RESAMPLED IMAGE WITH ISOTROPIC SPACING
def isotropic_resample(image, new_spacing=[1.0, 1.0, 1.0]):
    #Get the original size and spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Output the original size and spacing
    print("Original Size: {}".format(original_size))
    print("Original Spacing: {}".format(original_spacing))

    # Calculate the new size and spacing
    new_size = [round(original_size[i] * (original_spacing[i] / new_spacing[i])) for i in range(3)]

    # Resample image with linear interpolation to acheive the new spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampled_image = resampler.Execute(image)

    # Output the new size and spacing
    print("New Size: {}".format(resampled_image.GetSize()))
    print("New Spacing: {}".format(resampled_image.GetSpacing()))

    return resampled_image




# FUNCTION RESAMPLES THE ENTIRE DATASET TO ISOTROPIC SPACING
def resample_dataset(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = sitk.ImageSeriesReader()
    writer = sitk.ImageFileWriter()

    # Get the list of patient folders in the input folder
    patient_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    # Loop through each patient folder
    for patient_folder in patient_folders:
        # Get the list of series folders in the patient folder
        series_folders = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, f))]

        # Loop through each series folder
        for series_folder in series_folders:
            # Get the list of CT scan folders in the series folder
            ct_scan_folders = [os.path.join(series_folder, f) for f in os.listdir(series_folder) if os.path.isdir(os.path.join(series_folder, f))]
        
            # Loop through each CT scan folder
            for ct_scan_folder in ct_scan_folders:

                # Read the slices into one image
                dicom_names = reader.GetGDCMSeriesFileNames(ct_scan_folder)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # Resample the image
                resampled_image = isotropic_resample(image)

                # Write the resampled image to the output folder
                patient_id = os.path.basename(patient_folder)
                series_id = os.path.basename(series_folder)
                ct_scan_id = os.path.basename(ct_scan_folder)
                output_filename = os.path.join(output_folder, patient_id + "_" + series_id + "_" + ct_scan_id + "_resampled.nii")
                writer.SetFileName(output_filename)
                writer.Execute(resampled_image)


resample_dataset("../../dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI", "../../dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI_resampled")