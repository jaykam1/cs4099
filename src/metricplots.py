from matplotlib import pyplot as plt
import seaborn as sns
import os

def save_plot(metric_name, metric_list_train, metric_list_val, plot_name, model_name, output_folder = "plots"):
    
    folder_path = os.path.join(output_folder, model_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    epochs = range(1, len(metric_list_train)+1)
    plt.plot(epochs, metric_list_train, linestyle='-', color='blue', label=f"Training {metric_name}")
    plt.plot(epochs, metric_list_val, linestyle='-', color='orange', label=f"Validation {metric_name}")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(metric_list_train) + 1, 5))
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} per Epoch")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(folder_path, f"{plot_name}.png")
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(confusion_matrix, title, model_name, output_folder = "plots"):

    folder_path = os.path.join(output_folder, model_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False ,xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)

    output_path = os.path.join(folder_path, f"{title}.png")
    plt.savefig(output_path)
    plt.close()

