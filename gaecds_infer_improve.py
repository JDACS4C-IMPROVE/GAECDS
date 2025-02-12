import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from model_params_def import infer_params

# [MODEL] Model-specific imports, as needed
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import torch.utils.data as Data
# Add the "code" directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from model import *
from process import *

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    # Set the device
    device = torch.device(params["cuda_name"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # CNN model
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])
    
    # MLP model
    mlp_modelpath = frm.build_model_path(
        model_file_name=params["mlp_model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])
    
    # GAE model
    gae_modelpath = frm.build_model_path(
        model_file_name=params["gae_model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])
    # --------------------------------------------------------------------
    # Load inference data (ML data)
    # --------------------------------------------------------------------
    # Load full response and drug features datasets
    node_drug = pd.read_csv(os.path.join(params["input_data_dir"], params["response_labels_file"]))
    features = pd.read_csv(os.path.join(params["input_data_dir"], params["drug_fingerprints_file"]))

    # Load test response and cell features datasets
    test_response = pd.read_hdf(os.path.join(params["input_data_dir"], test_data_fname), key="response")
    test_cell_features = pd.read_hdf(os.path.join(params["input_data_dir"], test_data_fname), key="cell_features")
    
    # Count positive and negative labels 
    test_positive = (test_response[params["y_col_name"]] == 1).sum()
    test_negative = (test_response[params["y_col_name"]] == 0).sum()
    print(f"Test - Positive: {test_positive}, Negative: {test_negative}")

    # Extract features
    drug1 = np.array(node_drug[params["drug_col_name_1"]])
    drug2 = np.array(node_drug[params["drug_col_name_2"]])
    drug_label = np.array(node_drug[params["y_col_name"]])

    # Preprocess cell features
    test_cell_features = test_cell_features.drop([params["canc_col_name"]], axis=1)
    test_cell_features = torch.tensor(test_cell_features.values, dtype=torch.float32).to(device)
    num_features = test_cell_features.shape[1]

    # Preprocess drug features
    node_feature = np.array(features)
    node_feature = torch.tensor(node_feature)
    num_drug_features = node_feature.shape[1]
    node_feature = torch.reshape(node_feature, (-1, num_drug_features))
    node_feature = node_feature.to(torch.float32).to(device)

    # Create adjacency matrix (unchanged from training)
    adj = torch.zeros((features.shape[0], features.shape[0]), dtype=torch.float32).to(device)
    adj = adj_create(drug1,drug2,drug_label,features.shape[0])
    adj = torch.tensor(adj)
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------
    # Initialize and load saved models
    gae_model = Model(num_drug_features, 256, 128).to(device)
    gae_model.load_state_dict(torch.load(gae_modelpath, map_location=device))
    gae_model.eval()

    mlp_model = MLP(in_feats=num_features, out_feats=128).to(device)
    mlp_model.load_state_dict(torch.load(mlp_modelpath, map_location=device))
    mlp_model.eval()

    cnn_model = CNN(128, 64, 32, 1).to(device)
    cnn_model.load_state_dict(torch.load(modelpath, map_location=device))
    cnn_model.eval()
    
    # GAE: Drug Graph Embeddings
    with torch.no_grad():
        _, drug_embeddings = gae_model(adj, node_feature)

    # MLP: Cell Embeddings
    with torch.no_grad():
        cell_embeddings = mlp_model(test_cell_features, "sigmoid")
    
    # Filter to test indices
    test_indices = test_response["split_index"].values
    test_drug1 = torch.tensor(drug1[test_indices], dtype=torch.long).to(device)
    test_drug2 = torch.tensor(drug2[test_indices], dtype=torch.long).to(device)
    drug_label_test = drug_label[test_indices]
    test_drug_label_tensor = torch.tensor(drug_label_test).reshape(-1, 1)

    test_cell_embeddings = cell_embeddings.cpu().numpy()
    test_drug_embeddings = drug_embeddings.cpu().numpy()

    # Crate CNN input
    cnn_input = new_matrix_with_cell(test_drug1.cpu().numpy(),
                                    test_drug2.cpu().numpy(),
                                    test_cell_embeddings,
                                    test_drug_embeddings)

    # Convert CNN input to tensor and reshape
    cnn_input = torch.tensor(cnn_input, dtype=torch.float32).to(device)
    cnn_input = cnn_input.view(-1, 128, 1, 1)
    
    # CNN: Synergy Predictions
    with torch.no_grad():
        y_pred = cnn_model(cnn_input)

    # Convert to numpy
    y_pred = y_pred.cpu().numpy().flatten()
    y_pred_binary = (y_pred >= params["positive_threshold"]).astype(int)
    y_true = test_drug_label_tensor.cpu().numpy().flatten()
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_true,
        y_pred=y_pred_binary,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=y_true,
            y_pred=y_pred_binary,
            y_prob=y_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="gaecds_original_params.txt",
        additional_definitions=infer_params,
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])