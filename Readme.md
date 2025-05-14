# Readme file for running the model
I am using Conda for the running of the code

# Brain MRI Preprocessing Pipeline (for Autoencoder Ensemble)

## Step 1: Setup Python Environment with Conda
- Create and activate a new environment
```
conda create -n brain_mri_env python=3.8 -y
conda activate brain_mri_env
```

- Install necessary packages
```
pip install -r requirements.txt
```

## Step 2: Prepare Dataset
- Download the t2 tar file from the link [IXI Dataset](https://brain-development.org/ixi-dataset/) and extract `.tar` file in a separate Data folder.
```
tar -xvf file_name.tar
```

## Step 3: Preprocess the `.nii.gz` Files
- Run Script to preprocess data and the data will be stored to the PreProcessedData folder with required, train, test and validate folders.
```
cd Utils
python preprocess.py
```

## Step 4: Visualize or Save Preprocessed Slices
- View `.npy` files using matplotlib, Run the following script using the command 
```
python viewFile.py
```

