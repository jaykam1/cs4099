# Lung Nodule Classification Project

The model weights can also be found at: https://huggingface.co/jaykam1/my-candidate-models/tree/main

You can run pip install -r requirements to install the required Python libraries.

As I use PyTorch, there is no need for containers, and GPU accelerated workloads can be run in a Python venv.

The LIDC-IDRI dataset is not included in the repo, as it contains medical data and is 125GB in size. It can be downloaded from here: https://www.cancerimagingarchive.net/collection/ lidc-idri/. Download this and place it in a separate folder.

I would recommend the following folder structure: A root directory containing a code and a dataset directory. Clone the github repo into a directory called src inside the code directory and download the dataset into the dataset directory. This mirrors my structure, so you do not have to change file paths in different files.

In src run python createnoduledataset.py noduledataset. This will save each of the individual nodules as an npy file.

To run a Neural Architecture Search, run python trainmodel2.py. This will run indefinetly, until you choose to stop it using Ctrl + C. All plots, metrics, and model checkpoints from your search will be saved into plots, savedmetrics, and savedmodels (you should create these folders before you run the search).

You can then use the utility functions in searchresultutilities.py to identify the ten best architectures according to whichever metric you please. Manually edit the architectures list in furthertrain.py to include which architectures you would like to be used for further training. You can then run python furthertrain.py to further train your chosen architectures for 700 further epochs. All metrics, plots, and model checkpoints will be stored in finaltrain savedmetrics, finaltrain plots, and finaltrain savedmodels. Once again, create these directories before you run the further training.

Finally, manually modify the architectures list in testresults.py to include your chosen archi- tectures before running python testresults.py. This program should output the performance metrics for your chosen models, as well as the number of parameters each model uses.
