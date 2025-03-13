#We use best performing model from the last 150 epochs as the final model for each architecture
import torch
from obtainlunasplit import get_folds
from torch.utils.data import DataLoader
from dataset import NoduleDataset
import os
import sys
from lungnet2 import LungNet
import numpy as np
import pandas as pd
import gc

def archstring_to_architecture(arch_string):
    arch_string_one = [block.split() for block in arch_string.replace('[', '').replace(']', '').replace(',', '').split('-')]
    architecture = []
    for block in arch_string_one:
        layers = []
        for layer in block:
            layers.append(int(layer))
        architecture.append(layers)
    return architecture

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_system(architectures, test_loader):

    all_predictions = [[] for _ in range(len(architectures))]
    all_labels = []

    with torch.no_grad():
        for i, arch_string in enumerate(architectures):
            checkpoint_path = f"finaltrain_savedmodels/{arch_string}checkpoint.pth"
            if not os.path.exists(checkpoint_path):
                print("Cant find model checkpoint")
                sys.exit(1)

            arch_string = arch_string[:-4]

            architecture = archstring_to_architecture(arch_string)
            model = LungNet(architecture).cuda()

            print(f"Model {i+1} has {count_parameters(model)} parameters")

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Set the model to evaluation mode
            model.eval()

            model_predictions = []

            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                model_predictions.extend(predictions.cpu().numpy())
                
                if i == 0:
                    all_labels.extend(labels.cpu().numpy())

                all_predictions[i] = model_predictions
            del model
            torch.cuda.empty_cache()
            gc.collect()
    return all_predictions, all_labels


def calculate_single_metrics(predictions, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(labels)):
        if predictions[i] == 1:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 0:
                tn += 1
            else:
                fn += 1 
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0.0
    return acc, sens, spec, prec, f1
                

# Load the test data
architectures = ['[16, 16, 16]-[16, 64, 64, 128]-[128]', 
                 '[32]-[128, 128, 128, 128, 128]-[128, 128]', 
                 #'[4, 8, 16, 16, 16]-[16]-[16, 32, 32]', final ensenble only uses 9 this one has lowest validation accuracy so removed
                 '[32]-[32, 128, 128, 128]-[128, 128, 128, 128]', 
                 '[8]-[16, 16, 16, 64]-[128]', 
                 '[8]-[8, 8]-[8, 16, 64, 64, 64, 128]', 
                 '[4, 4, 4]-[4]-[16, 16, 32, 32, 64]', 
                 '[8, 8, 8, 64]-[128]-[128, 128, 128, 128]', 
                 '[64]-[64, 64, 128]-[128, 128]',
                 '[4, 8, 8]-[8]-[8, 16, 16, 128]']

#Get best epoch for each model from the last 150 epochs
def get_best_epoch_architecture(architectures):
    best_epoch_architectures = []
    for arch_string in architectures:
        arch_df = pd.read_csv(f"finaltrain_savedmetrics/{arch_string}_metrics.csv")
        best_epoch = 0
        best_acc = 0
        for epoch in range(549, 700, 50):
            best_acc = max(best_acc, arch_df['Validation Accuracy'].iloc[epoch])
            if arch_df['Validation Accuracy'].iloc[epoch] == best_acc:
                best_epoch = epoch+1
        print(f"Best epoch for {arch_string} is {best_epoch} with accuracy {best_acc}")
        best_epoch_architecture = f"{arch_string}_{best_epoch}"
        best_epoch_architectures.append(best_epoch_architecture)
    return best_epoch_architectures

best_epoch_architectures = get_best_epoch_architecture(architectures)

'''
best_epoch_architectures = ['[16, 16, 16]-[16, 64, 64, 128]-[128]_550', 
                            '[32]-[128, 128, 128, 128, 128]-[128, 128]_700', 
                            '[32]-[32, 128, 128, 128]-[128, 128, 128, 128]_650', 
                            '[8]-[16, 16, 16, 64]-[128]_700', 
                            '[8]-[8, 8]-[8, 16, 64, 64, 64, 128]_700', 
                            '[4, 4, 4]-[4]-[16, 16, 32, 32, 64]_700', 
                            '[8, 8, 8, 64]-[128]-[128, 128, 128, 128]_600', 
                            '[64]-[64, 64, 128]-[128, 128]_650']
                            '[4, 8, 8]-[8]-[8, 16, 16, 128]_650']
'''

folds = get_folds()
test_ids = folds[9]
test_dataset = NoduleDataset(nodule_dir="noduledataset2", labels_file="noduledataset2/labels.csv", patient_ids=test_ids)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
all_predictions, all_labels = test_system(best_epoch_architectures, test_loader)

assert all(len(pred) == len(all_labels) for pred in all_predictions), "Prediction and label dimensions are off"

ensemble_predictions = []
for i in range(len(all_labels)):
    positive_decisions = 0
    for j in range(len(all_predictions)):
        if all_predictions[j][i] == 1:
            positive_decisions += 1
    ensemble_predictions.append(1 if positive_decisions >= np.floor(len(all_predictions) / 2) else 0)

'''
CODE TO COMPARE PREDICTIONS OF ENSEMBLE AND MODEL 7
print("Ensemble predictions")
print(ensemble_predictions)
print("Model 7 predictions")
print(list(map(int, all_predictions[6])))
'''

'''
CODE TO GET TPS, TNS, FPS, FNS OF ENSEMBLE, MODEL 7, AND MODEL 3
ensemble_tp, ensemble_tn, ensemble_fp, ensemble_fn = 0, 0, 0, 0
model7_tp, model7_tn, model7_fp, model7_fn = 0, 0, 0, 0
model3_tp, model3_tn, model3_fp, model3_fn = 0, 0, 0, 0
for i in range(len(all_labels)):
    if ensemble_predictions[i] == 1:
        if all_labels[i] == 1:
            ensemble_tp += 1
        else:
            ensemble_fp += 1
    else:
        if all_labels[i] == 0:
            ensemble_tn += 1
        else:
            ensemble_fn += 1

    if all_predictions[6][i] == 1:
        if all_labels[i] == 1:
            model7_tp += 1
        else:
            model7_fp += 1
    else:
        if all_labels[i] == 0:
            model7_tn += 1
        else:
            model7_fn += 1

    if all_predictions[2][i] == 1:
        if all_labels[i] == 1:
            model3_tp += 1
        else:
            model3_fp += 1
    else:
        if all_labels[i] == 0:
            model3_tn += 1
        else:
            model3_fn += 1
print("Ensemble TP, TN, FP, FN")
print(ensemble_tp, ensemble_tn, ensemble_fp, ensemble_fn)
print("Model 7 TP, TN, FP, FN")
print(model7_tp, model7_tn, model7_fp, model7_fn)
print("Model 3 TP, TN, FP, FN")
print(model3_tp, model3_tn, model3_fp, model3_fn)
'''


'''
CODE TO GET METRICS OF ALL 10 MODELS
'''
accs = []
senss = []
specs = []
precs = []
f1s = []


for i, model_predictions in enumerate(all_predictions):
    acc, sens, spec, prec, f1 = calculate_single_metrics(model_predictions, all_labels)
    print(f"Model {i+1}")
    print(f"Accuracy: {acc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"Precision: {prec}")
    print(f"F1: {f1}")
    accs.append(acc)
    senss.append(sens)
    specs.append(spec)
    precs.append(prec)
    f1s.append(f1)

print("Average")
print(f"Accuracy: {np.mean(accs)}")
print(f"Sensitivity: {np.mean(senss)}")
print(f"Specificity: {np.mean(specs)}")
print(f"Precision: {np.mean(precs)}")
print(f"F1: {np.mean(f1s)}")



acc, sens, spec, prec, f1 = calculate_single_metrics(ensemble_predictions, all_labels)
print("Ensemble")
print(f"Accuracy: {acc}")
print(f"Sensitivity: {sens}")
print(f"Specificity: {spec}")
print(f"Precision: {prec}")
print(f"F1: {f1}")
