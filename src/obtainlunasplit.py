import pylidc as pl

# This function takes a scan ID and returns the corresponding patient ID
def scan_to_patient_id(scan_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == scan_id).first()
    if not scan:
        return "Scan not found"
    patient_id = int(scan.patient_id[10:])
    return patient_id

# This function takes a file name as input and returns a list of corresponding patient IDs
def file_to_patient_list(input_file):
    patient_list = []
    with open(input_file, 'r') as f_in:
        for i, line in enumerate(f_in):
            if i % 4 == 0:
                scan_id = line[:-5]
                patient_id = scan_to_patient_id(scan_id)
                patient_list.append(patient_id)
    return patient_list

# This function returns a dictionary mapping fold numbers to lists of patient IDs
# The folds are as per the LUNA16 dataset
def get_folds():
    fold_map = {}
    for i in range(10):
        patients_i = file_to_patient_list('uncleanfolds/fold{}.txt'.format(i))
        fold_map[i] = patients_i
    return fold_map

