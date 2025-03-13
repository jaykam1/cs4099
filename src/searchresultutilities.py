import os
import pandas as pd

# Get the number of architectures explored
def num_architectures():
    return len(os.listdir('savedmodels'))

# Get the best architecture according to a specific metric or average of metrics
def best_architecture(metric_name):
    best_architecture = None
    best_metric = None
    for architecture in os.listdir('savedmetrics'):
        all_metrics = pd.read_csv(os.path.join('savedmetrics', architecture))
        value = None
        if metric_name == 'All':
            val_acc = all_metrics['Validation Accuracy'].iloc[-1]
            val_loss = all_metrics['Validation Loss'].iloc[-1]
            val_sens = all_metrics['Validation Sensitivity'].iloc[-1]
            val_spec = all_metrics['Validation Specificity'].iloc[-1]
            val_prec = all_metrics['Validation Precision'].iloc[-1]
            val_f1 = all_metrics['Validation F1'].iloc[-1]
            value = (val_acc + val_loss + val_sens + val_spec + val_prec + val_f1) / 6
        else:
            value = all_metrics[metric_name].iloc[-1]
        if best_metric is None or value > best_metric:
            best_metric = value
            best_architecture = architecture
    return best_architecture, best_metric


# Get the best ten architectures according to a specific metric or average of metrics
def best_ten_architectures(metric_name):
    best_architectures = []
    best_metrics = []
    for filename in os.listdir('savedmetrics'):
        architecture = filename[:-12]
        all_metrics = pd.read_csv(os.path.join('savedmetrics', filename))
        value = None
        if metric_name == 'All':
            val_acc = all_metrics['Validation Accuracy'].iloc[-1]
            val_loss = all_metrics['Validation Loss'].iloc[-1]
            val_sens = all_metrics['Validation Sensitivity'].iloc[-1]
            val_spec = all_metrics['Validation Specificity'].iloc[-1]
            val_prec = all_metrics['Validation Precision'].iloc[-1]
            val_f1 = all_metrics['Validation F1'].iloc[-1]
            value = (val_acc + val_loss + val_sens + val_spec + val_prec + val_f1) / 6
        else:
            value = all_metrics[metric_name].iloc[-1]
        if len(best_architectures) < 10:
            best_architectures.append(architecture)
            best_metrics.append(value)
        else:
            min_index = best_metrics.index(min(best_metrics))
            if value > best_metrics[min_index]:
                best_architectures[min_index] = architecture
                best_metrics[min_index] = value
    return best_architectures, best_metrics

# Get the metrics of a specific architecture
def architecture_metrics(architecture_string):
    path = os.path.join('savedmetrics', architecture_string + '_metrics.csv')
    metric_dict = {}
    all_metrics = pd.read_csv(path)
    metric_dict['Validation Accuracy'] = all_metrics['Validation Accuracy'].iloc[-1]
    metric_dict['Validation Loss'] = all_metrics['Validation Loss'].iloc[-1]
    metric_dict['Validation Sensitivity'] = all_metrics['Validation Sensitivity'].iloc[-1]
    metric_dict['Validation Specificity'] = all_metrics['Validation Specificity'].iloc[-1]
    metric_dict['Validation Precision'] = all_metrics['Validation Precision'].iloc[-1]
    metric_dict['Validation F1'] = all_metrics['Validation F1'].iloc[-1]
    return metric_dict


def main():
    #print(num_architectures())
    #print(best_architecture('Validation Accuracy'))
    print(best_ten_architectures('Validation Accuracy'))
    #print(architecture_metrics('[16]-[16, 16, 32]-[64, 128, 128]'))

if __name__ == '__main__':
    main()
    
