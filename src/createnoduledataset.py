from initialpreprocess import get_all_nodules
import numpy as np
import sys

def save_nodules(dir):
    nodules = get_all_nodules()
    f = open(dir + '/labels.csv', 'w')
    f.write('nodule_id,patient_id,malignancy,malignancy_truth\n')
    for index, (nodule, patient_id, malignancy, malignancy_truth) in enumerate(nodules):
        np.save("{0}/{1}.npy".format(dir, index), nodule)
        labels = "{0},{1},{2},{3}\n".format(index, patient_id, malignancy, malignancy_truth)
        f.write(labels)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        save_nodules(sys.argv[1])
    else:
        print("Run createnoduledataset.py <output directory>")
