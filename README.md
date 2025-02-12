# IMPROVE - GAECDS: Drug Synergy Prediction
This is the IMPROVE implementation of the original model with original data.

## Dependencies and Installation
### Conda Environment
```
conda env create -f environment_improve.yml
conda activate gaecds_improve
```

### Clone Repository
```
git clone https://github.com/JDACS4C-IMPROVE/GAECDS.git
cd GAECDS
git checkout IMPROVE-original
```

### Clone IMPROVE Repository
```
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop # default branch
cd ..
```

### Download Original Data

The original data files necessary for this implementation are provided in this repository. Please refer to the `data/` directory to access the original data files.

## Running the Model

### 1. Activate the conda environment
```
conda activate gaecds_improve
```

### 2. Set environment variables
```
export PYTHONPATH=$PYTHONPATH:/your/path/to/IMPROVE
```

### 3. Prepare dataset in IMPROVE format
```
python prepare_drugcombo_data.py
```
This script prepares the authors' data from DrugComb and reformats them to fit the IMPROVE framework. The processed data is saved in the `improve_original` folder, following the required structure with `y_data`, `x_data`, and `splits` folders.

### 4. Preprocess raw data to construct model input data (ML data)
```
python gaecds_preprocess_improve.py --input_dir improve_original/ --output_dir original/ml_data
```
Preprocesses the IMPROVE-formatted data and creates train, validation (val), and test datasets.

Generates:

 * three model input data files: `train_data.h5`, `val_data.h5`, `test_data.h5`
 * three tabular data files, each containing the drug response values (i.e. Loewe) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`
 * two tabular data files needed for training the GAE model: `drug_ecfp4_nbits300.csv`, `response_all_label_only.csv` 

*Note: The binary classification labels generated in this script are based off the Loewe scores.*

### 5. Train model
```
python gaecds_train_improve.py --input_dir original/ml_data --output_dir original/out_models
```
Trains the GAECDS model using the model's input data: `train_data.h5` (training), `val_data.h5` (for early stopping).

Generates:

 * trained models: `gae_model.pt`, `mlp_model.pt`,  `model.pt`
 * predictions on val data (tabular data): `val_y_data_predicted.csv`
 * prediction performance scores on val data: `val_scores.json`

GAECDS generates three models:
1. Graph Autoencoder (GAE) model (`gae_model.pt`), which applies GCN methods to encode drug combinations by treating drugs as nodes and synergistic relations as edges, then decodes them via matrix factorization.
2. Multi-Layer Perceptron (MLP) model (`mlp_model.pt`), which processes cell-line features to incorporate cellular context into the prediction.
3. Convolutional Neural Network (CNN) model (`model.pt`), which takes both the GAE-generated latent vectors and MLP-processed cell-line features to predict synergistic scores.

 *Note: The training process occurs within a nested loop, where the GAE model is trained for 5 epochs in the original code, and for each GAE epoch, both the CNN and MLP models are further trained together for 200 epochs. This implementation removes the cross-fold validation, instead training on a single dataset split based on the IMPROVE framework. In the original implementation, cross-fold validation was used, but the GNN and MLP models were not reinitialized between folds, meaning that weights carried over across different validation splits. Additionally, the original code did not implement early stopping or save the best models. As a result, the model's performance may depend on the specific data split or changing the hyperparameters for early stopping.*

### 6. Run inference on test data with the trained model
```
python gaecds_infer_improve.py --input_data_dir original/ml_data --input_model_dir original/out_models --output_dir original/out_infer --calc_infer_scores True
```

Evaluates the performance on a test dataset, `test_data.h5`, with the trained models.

Generates:

 * predictions on test data (tabular data): `test_y_data_predicted.csv`
 * prediction performance scores on test data: `test_scores.json`

## References

Original GitHub: https://github.com/juneli126/GAECDS

Original Paper: https://link.springer.com/article/10.1007/s12539-023-00558-y

If you use this repository in your research or projects, please cite the original work:
```   
Li H, Zou L, Kowah JAH, et al. Predicting Drug Synergy and Discovering New Drug Combinations Based on a Graph Autoencoder and Convolutional Neural Network. Interdiscip Sci. 2023;15(2):316-330. doi:10.1007/s12539-023-00558-y
```

