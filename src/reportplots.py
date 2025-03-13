import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def plot_metric_for_all_architectures(metric_name):
    x_values = []
    for architecture in os.listdir('savedmetrics'):
        all_metrics = pd.read_csv(os.path.join('savedmetrics', architecture))
        x_values.append(all_metrics[metric_name].iloc[-1])

    x_values = [x * 100 for x in x_values]  
    plt.figure(figsize=(10, 2))
    plt.hist(x_values, bins=20, color ='orange')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylabel('Number of architectures')
    plt.title(f'{metric_name} (%) for all architectures')

    # Save the plot
    plt.savefig(f'reportplots/{metric_name}_for_all_architectures.png')

def plot_further_train_metrics(architecture, metric_name):
    x_values_train = []
    x_values_validation = []
    all_metrics = pd.read_csv(os.path.join('finaltrain_savedmetrics', f"{architecture}_metrics.csv"))
    x_values_train = all_metrics[metric_name]
    x_values_validation = all_metrics[f'Validation {metric_name}']
    x_values_train = [x * 100 for x in x_values_train]
    x_values_validation = [x * 100 for x in x_values_validation]

    epochs = np.arange(1, 701)  
    window_size = 10
    smoothed_train = np.convolve(x_values_train, np.ones(window_size)/window_size, mode='valid')
    smoothed_validation = np.convolve(x_values_validation, np.ones(window_size)/window_size, mode='valid')

    smoothed_epochs = epochs[:len(smoothed_train)]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, x_values_train, color='blue', alpha=0.2, label=f'Train {metric_name} (Raw)')
    plt.plot(epochs, x_values_validation, color='orange', alpha=0.2, label=f'Validation {metric_name} (Raw)')
    plt.plot(smoothed_epochs, smoothed_train, color='blue', linewidth=2, label=f'Train {metric_name} (Smoothed)')
    plt.plot(smoothed_epochs, smoothed_validation, color='orange', linewidth=2, label=f'Validation {metric_name} (Smoothed)')

    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'Training vs Validation {metric_name} for ' + architecture)
    plt.legend()
    plt.grid(alpha=0.3)

    # Show the plot
    plt.savefig(f'reportplots/{metric_name}_for_{architecture}.png')

def plot_lightweight_model_comparison():
    models = ["My Ensemble", "NASLung Ensemble", "Model 5", "AE-DPN", "DeepLung"]
    accuracy = [90.91, 90.77, 90.91, 90.24, 90.44]
    sensitivity = [84.00, 85.37, 84.00, 92.04, 81.42]
    specificity = [96.67, 95.04, 96.67, 88.94, np.nan]
    f1_score = [89.36, 89.29, 89.36, 90.45, np.nan]

    metrics = ["Accuracy", "Sensitivity", "Specificity", "F1 Score"]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15  # Width of bars

    for i, model in enumerate(models):
        values = [accuracy[i], sensitivity[i], specificity[i], f1_score[i]]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Performance (%)")
    ax.set_title("Performance Metrics Comparison Across Lightweight Models")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(80, 100)
    plt.savefig("reportplots/lightweight_model_comparison.png")

def plot_f1_score_against_parameters():
    models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8", "Model 9",
               "NASLung", "AE-DPN", "DeepLung"]
    params = [2.44, 6.47, 6.53, 0.97, 1.29, 0.30, 4.97, 3.60, 0.54, 16.84, 678.69, 141.57] 
    f1_scores = [80.00, 85.71, 89.80, 83.72, 89.36, 86.96, 89.36, 85.71, 87.50, 89.29, 90.45, np.nan]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(params[:-1], f1_scores[:-1], color="blue", marker='x', label="Candidate Models", s=80)  
    ax.scatter(params[-3:], f1_scores[-3:], color="orange",  marker = 'x', label="SOTA Lightweight-Models", s=100)  
    ax.set_xscale("log")  # Log scale for better visualization

    for i, txt in enumerate(models):
        ax.annotate(txt, (params[i], f1_scores[i]), fontsize=10, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Number of Parameters (Millions)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score vs. Number of Parameters for Lightweight Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("reportplots/f1_score_vs_parameters.png")

def plot_accuracy_against_parameters():
    models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8", "Model 9",
               "NASLung", "AE-DPN", "DeepLung"]
    params = [2.44, 6.47, 6.53, 0.97, 1.29, 0.30, 4.97, 3.60, 0.54, 16.84, 678.69, 141.57] 
    f1_scores = [83.64, 87.27, 90.91, 87.27, 90.91, 89.10, 90.91, 87.27, 89.10, 90.77, 90.24, 90.44]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(params[:-1], f1_scores[:-1], color="blue", marker='x', label="Candidate Models", s=80)  
    ax.scatter(params[-3:], f1_scores[-3:], color="orange",  marker = 'x', label="SOTA Lightweight-Models", s=100)  
    ax.set_xscale("log")  # Log scale for better visualization

    for i, txt in enumerate(models):
        ax.annotate(txt, (params[i], f1_scores[i]), fontsize=10, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Number of Parameters (Millions)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Number of Parameters for Lightweight Models")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("reportplots/accuracy_vs_parameters.png")

def plot_test_confusion_matrices():
    models = ["Ensemble", "Model 7", "Model 3"]
    confusion_matrices = [[[21, 4], [1, 29]], [[21, 4], [1, 29]], [[22, 3], [2, 28]]]
    for i in range(len(models)):
        confusion_matrix = confusion_matrices[i]
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False ,xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Test Set Confusion Matrix for {models[i]}")
        plt.savefig(f"reportplots/{models[i]}_testset_confusion_matrix.png")
        plt.close()

if __name__ == '__main__':
    '''
    plot_metric_for_all_architectures('Validation Accuracy')
    plot_metric_for_all_architectures('Validation Loss')
    plot_metric_for_all_architectures('Validation Sensitivity')
    plot_metric_for_all_architectures('Validation Specificity')
    plot_metric_for_all_architectures('Validation Precision')
    plot_metric_for_all_architectures('Validation F1')

    model1_architecture = '[16, 16, 16]-[16, 64, 64, 128]-[128]'
    plot_further_train_metrics(model1_architecture, 'Accuracy')
    plot_further_train_metrics(model1_architecture, 'Loss')
    plot_further_train_metrics(model1_architecture, 'Sensitivity')
    plot_further_train_metrics(model1_architecture, 'Specificity')
    plot_further_train_metrics(model1_architecture, 'Precision')
    plot_further_train_metrics(model1_architecture, 'F1')
    
    
    plot_f1_score_against_parameters()
    plot_accuracy_against_parameters()
    '''
    plot_lightweight_model_comparison()
    plot_test_confusion_matrices()
