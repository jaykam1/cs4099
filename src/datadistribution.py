from matplotlib import pyplot as plt
from initialpreprocess import patient_to_scan_id, median_rating
import pandas as pd
import pylidc as pl


'''
Plot a bar chart showing distribution of nodule sizes
<5 mm - 
5-12mm - most difficult to classify
>12mm
'''
def plot_nodule_size_distribution():
    nodules = get_nodules()
    less_than_5 = 0
    five_to_twelve = 0
    greater_than_twelve = 0
    for nodule in nodules:
        size = nodule[0]
        if size < 5:
            less_than_5 += 1
        elif size < 12:
            five_to_twelve += 1
        else:
            greater_than_twelve += 1
    plt.bar(['<5mm', '5-12mm', '>12mm'], [less_than_5, five_to_twelve, greater_than_twelve])
    plt.title('Distribution of Nodule Sizes')
    plt.xlabel('Size')
    plt.ylabel('Number of Nodules')
    plt.savefig('distributionplots/nodule_size_distribution.png')

'''
Plot a bar chart showing distribution of nodule types
Benign
Malignant
'''
def plot_nodule_type_distribution():
    nodules = get_nodules()
    benign = 0
    malignant = 0
    for nodule in nodules:
        if nodule[1]:
            malignant += 1
        else:
            benign += 1
    plt.bar(['Benign', 'Malignant'], [benign, malignant])
    plt.title('Distribution of Nodule Types')
    plt.xlabel('Type')
    plt.ylabel('Number of Nodules')
    plt.savefig('distributionplots/nodule_type_distribution.png')

def get_nodules():
    df = pd.read_csv('list3.2.csv')
    query = pl.query(pl.Scan)
    patient_to_scan = patient_to_scan_id(query)
    # Nodules of form (size, malignancy_truth)
    nodules = []
    for index, row in df.iterrows():
        if row['eq. diam.'] > 30:
            continue
        patient_id = row['case']
        if patient_id not in patient_to_scan:
            continue
        scan_id = patient_to_scan[patient_id]
        scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
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
        nodules.append((row['eq. diam.'], malignancy_truth))
    return nodules

def main():
    #plot_nodule_size_distribution()
    plot_nodule_type_distribution()

if __name__ == '__main__':
    main()

