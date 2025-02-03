import pandas as pd
import pylidc as pl
import pydicom
import os 
import numpy as np

def get_a_file(path):
    files = os.listdir(path)
    dicom_files = [(path + "/" + file) for file in files if file.endswith('.dcm')]
    if len(dicom_files) < 1:
        return None
    return pydicom.dcmread(dicom_files[0])

def get_an_annotation(annotations):
    return annotations[0]

def convert_to_hu(vol, slope, intercept):
    return (vol * slope + intercept)

def bound_hu(vol):
    return np.clip(vol, -1000, 400)

def normalise_hu(vol):
    mean = np.mean(vol)
    std = np.std(vol)
    return (vol - mean) / std if std != 0 else vol

def average_centroid(annotations):
    centroids = [annotation.centroid for annotation in annotations]
    return np.mean(centroids, axis=0)

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

def median_rating(annotations):
    ratings = [annotation.malignancy for annotation in annotations]
    return np.median(ratings).astype(int)

def get_all_nodules():
    df = pd.read_csv('list3.2.csv')
    query = pl.query(pl.Scan)
    nodules = []
    patient_to_scan = patient_to_scan_id(query) 
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
        hu_normalised = normalise_hu(hu_bound)
        
        nodules.append((hu_normalised, patient_id, malignancy, malignancy_truth)) 
        
    return nodules

