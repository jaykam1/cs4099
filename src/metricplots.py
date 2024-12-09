from matplotlib import pyplot as plt
import seaborn as sns

def save_plot(metric_name, metric_list_train, metric_list_val, plot_name):
    epochs = range(1, len(metric_list_train)+1)
    plt.plot(epochs, metric_list_train, marker='o', color='b', label=f"Training {metric_name}")
    plt.plot(epochs, metric_list_val, marker='o', color='r', label=f"Validation {metric_name}")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(metric_list_train) + 1, 5))
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{plot_name}.png")
    plt.close()


def plot_confusion_matrix(confusion_matrix, title):
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False ,xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.close()

