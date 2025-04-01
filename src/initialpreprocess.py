import pandas as pd
import pylidc as pl
import pydicom
import os 
import numpy as np
from obtainlunasplit import get_folds

# Reads the dicom file at the given path
def get_a_file(path):
    files = os.listdir(path)
    dicom_files = [(path + "/" + file) for file in files if file.endswith('.dcm')]
    if len(dicom_files) < 1:
        return None
    return pydicom.dcmread(dicom_files[0])

# Gets the first annotation from the list of annotations
def get_an_annotation(annotations):
    return annotations[0]

# Converts a volume from arbitrary units to Hounsfield Units
def convert_to_hu(vol, slope, intercept):
    return (vol * slope + intercept)

# Bounds the Hounsfield Units to a range of -1000 to 400 (from air to bone)
def bound_hu(vol):
    return np.clip(vol, -1000, 400)

# Normalises the Hounsfield Units using the mean and standard deviation of the training set
def normalise_hu(vol, train_mean, train_std):
    return (vol - train_mean) / train_std

# Gets the average centroid of a list of annotation, used to find the centroid of a nodule
def average_centroid(annotations):
    centroids = [annotation.centroid for annotation in annotations]
    return np.mean(centroids, axis=0)

# Gets the map of patient ids to scan ids
def patient_to_scan_id(query):
    pts_map = {}
    for scan in query:
        scan_path = scan.get_path_to_dicom_files()
        dicom_data = get_a_file(scan_path)
        if dicom_data:
            pts_map[int(dicom_data.PatientID[10:])] = scan.id
        else:
            continue
    return pts_map 

# Gets the median malignancy rating of a list of annotations
def median_rating(annotations):
    ratings = [annotation.malignancy for annotation in annotations]
    return np.median(ratings).astype(int)


def get_all_nodules():
    df = pd.read_csv('list3.2.csv')
    query = pl.query(pl.Scan)
    # Nodules of form (nodule_volume, patient_id, malignancy_rating, malignancy_truth)
    nodules = []
    patient_to_scan = patient_to_scan_id(query) 
    # Get the folds for the LUNA16 dataset
    folds = get_folds()
    train_ids = [pid for fold in range(0, 8) for pid in folds[fold]]

    #First pass to store all training HU values for working out mean and std so we can normalise
    train_hu_values = []

    # Iterates through each row (each nodule) in the dataframe
    for index, row in df.iterrows():
        
        # Skip nodules that are larger than 30mm
        if row['eq. diam.'] > 30:
            continue
        
        patient_id = row['case']

        if patient_id not in patient_to_scan:
            continue
        
        # Uses the DICOM data to get the nodule data, intercept, and slope
        scan_id = patient_to_scan[patient_id]
        scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
        dicom_data = get_a_file(scan.get_path_to_dicom_files())
        intercept = dicom_data.RescaleIntercept
        slope = dicom_data.RescaleSlope
        
        nodule_names = row[8:][row[8:].notnull()].values
        
        # Skip nodules that have less than 3 annotations
        if len(nodule_names) < 3:
            continue

        annotations = [a for a in scan.annotations if a._nodule_id in nodule_names]

        if not annotations:
            continue

        # Skip nodules that have a median malignancy rating of 3
        malignancy = median_rating(annotations)
        malignancy_truth = malignancy > 3
        if malignancy == 3:
            continue

        # Skip nodules that have a maximum bounding box size of less than 32mm
        annotations = [a for a in annotations if max(a.bbox_dims(pad=1)) <= 31]

        if not annotations:
            continue

        # Resample so nodule has an isotropic resolution
        a = get_an_annotation(annotations)
        centroid = average_centroid(annotations)
        vol, _ = a.uniform_cubic_resample(side_length=31)
        
        # Convert to Hounsfield Units and bound the values
        hu_vol = convert_to_hu(vol, slope, intercept)
        hu_bound = bound_hu(hu_vol)

        if patient_id in train_ids:
            train_hu_values.append(hu_bound)
    
    # Get the mean and std of the training set
    train_hu_values = np.concatenate([v.flatten() for v in train_hu_values])
    train_mean = train_hu_values.mean()
    train_std = train_hu_values.std()

    #Second pass to normalise and store nodules - do same steps as above but this time normalise the nodule 
    # and store it in the nodules list and return it
    for index, row in df.iterrows():

        if row['eq. diam.'] > 30:
            continue

        patient_id = row['case']

        if patient_id not in patient_to_scan:
            continue

        scan_id = patient_to_scan[patient_id]
        scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
        dicom_data = get_a_file(scan.get_path_to_dicom_files())
        intercept = dicom_data.RescaleIntercept
        slope = dicom_data.RescaleSlope

        nodule_names = row[8:][row[8:].notnull()].values

        if len(nodule_names) < 3:
            continue

        annotations = [a for a in scan.annotations if a._nodule_id in nodule_names]

        if not annotations:
            continue

        malignancy = median_rating(annotations)
        malignancy_truth = malignancy > 3
        if malignancy == 3:
            continue

        annotations = [a for a in annotations if max(a.bbox_dims(pad=1)) <= 31]

        if not annotations:
            continue

        a = get_an_annotation(annotations)
        centroid = average_centroid(annotations)
        vol, _ = a.uniform_cubic_resample(side_length=31)
        
        hu_vol = convert_to_hu(vol, slope, intercept)
        hu_bound = bound_hu(hu_vol)

        # Apply the normalisation
        hu_normalised = normalise_hu(hu_bound, train_mean, train_std)
        
        nodules.append((hu_normalised, patient_id, malignancy, malignancy_truth)) 
        
    return nodules


