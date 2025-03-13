from dataset import NoduleDataset
import torch
import torch.nn as nn
from obtainlunasplit import get_folds
from torch.utils.data import DataLoader
from metricplots import save_plot, plot_confusion_matrix
import numpy as np
import gc
import pandas as pd
from lungnet2 import LungNet
import random
from torch.autograd import Variable
import os
import sys

def archstring_to_architecture(arch_string):
    arch_string_one = [block.split() for block in arch_string.replace('[', '').replace(']', '').replace(',', '').split('-')]
    architecture = []
    for block in arch_string_one:
        layers = []
        for layer in block:
            layers.append(int(layer))
        architecture.append(layers)
    return architecture

'''
The purpose of this program is to train the 10 best architectures from the NAS search
for 700 epochs each. 
'''

def train_model(model, num_epochs, criterion, optimizer, name):

    val_dataset = NoduleDataset(nodule_dir="noduledataset2", labels_file="noduledataset2/labels.csv", patient_ids=val_ids)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
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

    last_epoch_tp = 0
    last_epoch_tn = 0
    last_epoch_fp = 0
    last_epoch_fn = 0

    #Make comments around funcitons and loops
    
    
    for epoch in range(num_epochs):
        '''
        Changed curriculum learning so still get some hard examples in the first half of training
        and get some easy examples in the second half of training
        This is to prevent model from overfitting to easy examples in the first half of training
        and to prevent model from overfitting to hard examples in the second half of training
        '''
        if epoch < 200:
            difficulty = "easy" if random.random() < 0.9 else "hard"
        elif epoch < 500:
            difficulty = "easy" if random.random() < 0.5 else "hard"
        else:
            difficulty = "easy" if random.random() < 0.1 else "hard"

        train_dataset = NoduleDataset(nodule_dir="noduledataset2", labels_file="noduledataset2/labels.csv", patient_ids=train_ids, augmentation=True, difficulty=difficulty)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels.long())
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs, 1)
            
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
            
            total_train += labels.size(0)
            if (tp + tn + fp + fn != total_train):
                print("Incorrectly counting tp/tn/fp/fn's")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        epoch_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        epoch_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        epoch_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        epoch_f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0.0

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

                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * inputs.size(0)

                _, predictions = torch.max(outputs, 1)

                val_tp += ((predictions == 1) & (labels == 1)).sum().item()
                val_tn += ((predictions == 0) & (labels == 0)).sum().item()
                val_fp += ((predictions == 1) & (labels == 0)).sum().item()
                val_fn += ((predictions == 0) & (labels == 1)).sum().item()
                
                total_val += labels.size(0)
                if (val_tp + val_tn + val_fp + val_fn != total_val):
                    print("Incorrectly counting validations tp/tn/fp/fn's")
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn) if (val_tp + val_tn + val_fp + val_fn) > 0 else 0.0
        val_epoch_sens = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        val_epoch_spec = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0.0
        val_epoch_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_epoch_f1 = (2 * val_tp) / ((2 * val_tp) + val_fp + val_fn) if ((2 * val_tp) + val_fp + val_fn) > 0 else 0.0

        val_loss_values.append(val_epoch_loss)
        val_accuracy_values.append(val_epoch_acc)
        val_sensitivity_values.append(val_epoch_sens)
        val_specificity_values.append(val_epoch_spec)
        val_precision_values.append(val_epoch_prec)
        val_f1_values.append(val_epoch_f1)

        print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}, Sensitivity: {val_epoch_sens:.4f}, Specificity: {val_epoch_spec:.4f}, Precision: {val_epoch_prec:.4f}, F1: {val_epoch_f1:.4f}\n")
        torch.cuda.empty_cache()

        if epoch == num_epochs - 1:
            last_epoch_tp = val_tp
            last_epoch_tn = val_tn
            last_epoch_fp = val_fp
            last_epoch_fn = val_fn

        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, f"finaltrain_savedmodels/{name}_{epoch+1}checkpoint.pth")

    validation_confusion_matrix = np.array([[int(last_epoch_tp), int(last_epoch_fn)], [int(last_epoch_fp), int(last_epoch_tn)]])
    plot_confusion_matrix(validation_confusion_matrix, "{}_Validation_Confusion_Matrix_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")

    save_plot("Loss", loss_values, val_loss_values, "{}_loss_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")
    save_plot("Accuracy", accuracy_values, val_accuracy_values, "{}_acc_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")
    save_plot("Sensitivity", sensitivity_values, val_sensitivity_values, "{}_sens_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")
    save_plot("Specificity", specificity_values, val_specificity_values, "{}_spec_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")
    save_plot("Precision", precision_values, val_precision_values, "{}_prec_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")
    save_plot("F1", f1_values, val_f1_values, "{}_f1_per_epoch_{}".format(name, num_epochs), name, output_folder="finaltrain_plots")

    #Write per epoch metrics to file
    metrics = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Loss": loss_values,
        "Accuracy": accuracy_values,
        "Sensitivity": sensitivity_values,
        "Specificity": specificity_values,
        "Precision": precision_values,
        "F1": f1_values,
        "Validation Loss": val_loss_values,
        "Validation Accuracy": val_accuracy_values,
        "Validation Sensitivity": val_sensitivity_values,
        "Validation Specificity": val_specificity_values,
        "Validation Precision": val_precision_values,
        "Validation F1": val_f1_values
    })
    metrics.to_csv(f"finaltrain_savedmetrics/{name}_metrics.csv", index=False)

    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()
 

# SPLIT DATA INTO TRAIN AND VALIDATION HERE
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
folds = get_folds()
train_ids = [pid for fold in range(0, 8) for pid in folds[fold]]  
val_ids = folds[8]
test_ids = folds[9]

batch_size = 8
lr = 0.0002
epoch = 700

architectures = ['[16, 16, 16]-[16, 64, 64, 128]-[128]', 
                 '[32]-[128, 128, 128, 128, 128]-[128, 128]', 
                 '[4, 8, 16, 16, 16]-[16]-[16, 32, 32]', 
                 '[32]-[32, 128, 128, 128]-[128, 128, 128, 128]', 
                 '[8]-[16, 16, 16, 64]-[128]', 
                 '[8]-[8, 8]-[8, 16, 64, 64, 64, 128]', 
                 '[4, 4, 4]-[4]-[16, 16, 32, 32, 64]', 
                 '[8, 8, 8, 64]-[128]-[128, 128, 128, 128]', 
                 '[64]-[64, 64, 128]-[128, 128]', 
                 '[4, 8, 8]-[8]-[8, 16, 16, 128]']

for arch_string in architectures:
    checkpoint_path = f"savedmodels/model-{arch_string}_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        print("Cant find model checkpoint")
        sys.exit(1)

    architecture = archstring_to_architecture(arch_string)
    model = LungNet(architecture).cuda()

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 20)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    print(f"Training model with architecture: {arch_string} from epoch {start_epoch}\n")
    train_model(model, epoch, criterion, optimiser, arch_string)

    del model, optimiser, criterion
    gc.collect()
    torch.cuda.empty_cache()
