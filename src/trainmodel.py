from lungnet import LungNet
from dataset import NoduleDataset
import torch
import torch.nn as nn
from obtainlunasplit import get_folds
from torch.utils.data import DataLoader
from metricplots import save_plot, plot_confusion_matrix
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, name):

    loss_values= []
    accuracy_values = []
    sensitivity_values = []
    specificity_values = []
    precision_values = []
    f1_values = []

    val_loss_values = []
    val_accuracy_values = []
    val_sensitivity_values = []
    val_specificity_values = []
    val_precision_values = []
    val_f1_values = []

    total_val_tp = 0
    total_val_tn = 0
    total_val_fp = 0
    total_val_fn = 0

    #Make comments artoind funcitons and loops
    for epoch in range(num_epochs):
        model.train()

        # Training Metrics
        running_loss = 0.0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total_train = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

            predictions = (torch.sigmoid(outputs.flatten()) > 0.5).float()
            tp += torch.logical_and(predictions == 1.0, labels == 1.0).sum().item()
            tn += torch.logical_and(predictions == 0.0, labels == 0.0).sum().item()
            fp += torch.logical_and(predictions == 1.0, labels == 0.0).sum().item()
            fn += torch.logical_and(predictions == 0.0, labels == 1.0).sum().item()
            
            total_train += labels.size(0)
            if (tp + tn + fp + fn != total_train):
                print("Incorrectly counting tp/tn/fp/fn's")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (tp + tn) / (tp + tn + fp + fn)
        epoch_sens = tp / (tp + fn)
        epoch_spec = tn / (tn + fp)
        epoch_prec = tp / (tp + fp)
        epoch_f1 = (2 * tp) / ((2 * tp) + fp + fn)

        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_acc)
        sensitivity_values.append(epoch_sens)
        specificity_values.append(epoch_spec)
        precision_values.append(epoch_prec)
        f1_values.append(epoch_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Sensitivity: {epoch_sens:.4f}, Specificity: {epoch_spec:.4f}, Precision: {epoch_prec:.4f}, F1: {epoch_f1:.4f}")
        
        model.eval()
        
        # Validation metrics        
        val_loss = 0.0
        val_tp = 0.0
        val_tn = 0.0
        val_fp = 0.0
        val_fn = 0.0
        total_val = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                loss = criterion(outputs.flatten(), labels)
                val_loss += loss.item() * inputs.size(0)

                predictions = (torch.sigmoid(outputs.flatten()) > 0.5).float()
                val_tp += torch.logical_and(predictions == 1.0, labels == 1.0).sum().item()
                val_tn += torch.logical_and(predictions == 0.0, labels == 0.0).sum().item()
                val_fp += torch.logical_and(predictions == 1.0, labels == 0.0).sum().item()
                val_fn += torch.logical_and(predictions == 0.0, labels == 1.0).sum().item()

                total_val_tp += val_tp
                total_val_tn += val_tn
                total_val_fp += val_fp
                total_val_fn += val_fn
                
                total_val += labels.size(0)
                if (val_tp + val_tn + val_fp + val_fn != total_val):
                    print("Incorrectly counting validations tp/tn/fp/fn's")
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn)
        val_epoch_sens = val_tp / (val_tp + val_fn)
        val_epoch_spec = val_tn / (val_tn + val_fp)
        val_epoch_prec = val_tp / (val_tp + val_fp)
        val_epoch_f1 = (2 * val_tp) / ((2 * val_tp) + val_fp + val_fn)

        val_loss_values.append(val_epoch_loss)
        val_accuracy_values.append(val_epoch_acc)
        val_sensitivity_values.append(val_epoch_sens)
        val_specificity_values.append(val_epoch_spec)
        val_precision_values.append(val_epoch_prec)
        val_f1_values.append(val_epoch_f1)

        print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}, Sensitivity: {val_epoch_sens:.4f}, Specificity: {val_epoch_spec:.4f}, Precision: {val_epoch_prec:.4f}, F1: {val_epoch_f1:.4f}\n")

    print("TOTAL VAL TP: {}".format(total_val_tp))
    print("TOTAL VAL TN: {}".format(total_val_tn))
    print("TOTAL VAL FP: {}".format(total_val_fp))
    print("TOTAL VAL FN: {}".format(total_val_fn))

    validation_confusion_matrix = np.array([[int(total_val_tp), int(total_val_fn)], [int(total_val_fp), int(total_val_tn)]])
    plot_confusion_matrix(validation_confusion_matrix, "{}_Validation_Confusion_Matrix_{}".format(name, num_epochs))

    save_plot("Loss", loss_values, val_loss_values, "{}_loss_per_epoch_{}".format(name, num_epochs))
    save_plot("Accuracy", accuracy_values, val_accuracy_values, "{}_acc_per_epoch_{}".format(name, num_epochs))
    save_plot("Sensitivity", sensitivity_values, val_sensitivity_values, "{}_sens_per_epoch_{}".format(name, num_epochs))
    save_plot("Specificity", specificity_values, val_specificity_values, "{}_spec_per_epoch_{}".format(name, num_epochs))
    save_plot("Precision", precision_values, val_precision_values, "{}_prec_per_epoch_{}".format(name, num_epochs))
    save_plot("F1", f1_values, val_f1_values, "{}_f1_per_epoch_{}".format(name, num_epochs))

# These are all just set to some value for now, will optimise hyperparameters later
#model1 = LungNet(num_classes=1).cuda()
#criterion = nn.BCEWithLogitsLoss()  
#optimizer = torch.optim.Adam(model1.parameters(), lr=0.00001)

# SPLIT DATA INTO TRAIN AND VALIDATION HERE
folds = get_folds()
train_ids = [pid for fold in range(0, 8) for pid in folds[fold]]  
val_ids = folds[8]
test_ids = folds[9]

train_dataset = NoduleDataset(nodule_dir="noduledataset", labels_file="noduledataset/labels.csv", patient_ids=train_ids, augmentation=True)
val_dataset = NoduleDataset(nodule_dir="noduledataset", labels_file="noduledataset/labels.csv", patient_ids=val_ids)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# NAS
block_space = [2, 3]
model_number = 0
for res3_num in block_space:
    for res4_num in block_space:
        for res5_num in block_space:
            model = LungNet(num_classes=1, res3_blocks=res3_num, res4_blocks=res4_num, res5_blocks=res5_num).cuda()
            criterion = nn.BCEWithLogitsLoss()  
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
            name_string = "model-{}-{}-{}".format(res3_num, res4_num, res5_num)
            train_model(model, train_loader, val_loader, num_epochs=10, criterion=criterion, optimizer=optimizer, name=name_string)

"""
With this base architecture this currently takes 7.5 minutes per epoch to train
So assuming around 60 epochs, will take 450 minutes to train for one architecture
To reduce to search time we will use a set of hyperparamaters given in one of the papers for the NAS and then will reoptimise the hyperparameters for 
the model(s) acheived by the NAS.
Assuming we want to search for a week (10,080 minutes) we can explore a search space of ~22 architectures
"""