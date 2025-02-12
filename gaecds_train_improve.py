import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from model_params_def import train_params

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
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    # Set the device
    device = torch.device(params["cuda_name"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    # CNN model
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])
    
    # MLP model
    mlp_modelpath = frm.build_model_path(
        model_file_name=params["mlp_model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])
    
    # GAE model
    gae_modelpath = frm.build_model_path(
        model_file_name=params["gae_model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])

    # --------------------------------------------------------------------
    # Load model input data (ML data) for train and val
    # --------------------------------------------------------------------
    # Load full response and drug features datasets
    node_drug = pd.read_csv(os.path.join(params["input_dir"], params["response_labels_file"]))
    features = pd.read_csv(os.path.join(params["input_dir"], params["drug_fingerprints_file"]))

    # Load train and val response and cell features datasets
    train_response = pd.read_hdf(os.path.join(params["input_dir"], train_data_fname), key="response")
    train_cell_features = pd.read_hdf(os.path.join(params["input_dir"], train_data_fname), key="cell_features")
    val_response = pd.read_hdf(os.path.join(params["input_dir"], val_data_fname), key="response")
    val_cell_features = pd.read_hdf(os.path.join(params["input_dir"], val_data_fname), key="cell_features")
    
    # Count positive and negative labels in training and validation datasets
    train_positive = (train_response[params["y_col_name"]] == 1).sum()
    train_negative = (train_response[params["y_col_name"]] == 0).sum()
    val_positive = (val_response[params["y_col_name"]] == 1).sum()
    val_negative = (val_response[params["y_col_name"]] == 0).sum()
    # Print results
    print(f"Train - Positive: {train_positive}, Negative: {train_negative}")
    print(f"Validation - Positive: {val_positive}, Negative: {val_negative}")

    # Preprocess drug interaction data and create adjacency matrix for GAE model
    drug1 = np.array(node_drug[params["drug_col_name_1"]])
    drug2 = np.array(node_drug[params["drug_col_name_2"]])
    drug_label = np.array(node_drug[params["y_col_name"]])
    adj = adj_create(drug1,drug2,drug_label,features.shape[0])
    adj = torch.tensor(adj)
    pos_weight = torch.tensor((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    labels = torch.reshape(adj, [-1])

    # Process train cell features
    train_cell_features = train_cell_features.drop([params["canc_col_name"]], axis=1)
    train_cell_features = torch.tensor(np.array(train_cell_features))
    num_features = train_cell_features.shape[1]
    train_cell_features = torch.reshape(train_cell_features, (-1, num_features))
    train_cell_features = train_cell_features.to(torch.float32)

    # Process validation cell features
    val_cell_features = val_cell_features.drop([params["canc_col_name"]], axis=1)
    val_cell_features = torch.tensor(np.array(val_cell_features))
    num_features_val = val_cell_features.shape[1]  # Get dynamically (should match train)
    val_cell_features = torch.reshape(val_cell_features, (-1, num_features_val))
    val_cell_features = val_cell_features.to(torch.float32)

    print("Train cell feature shape:", train_cell_features.shape)
    print("Validation cell feature shape:", val_cell_features.shape)
    
    # Process drug features
    node_feature = np.array(features)
    node_feature = torch.tensor(node_feature)
    num_drug_features = node_feature.shape[1]
    node_feature = torch.reshape(node_feature, (-1, num_drug_features))
    node_feature = node_feature.to(torch.float32)
    print("Drug feature shape:", node_feature.shape)
    
    # --------------------------------------------------------------------
    # Prepare models
    # --------------------------------------------------------------------
    # CNN model
    net = CNN(128,64,32,1).to(device)
    
    # MLP model
    cell_mlp = MLP(in_feats=num_features, out_feats=128).to(device)

    opt_cnn = torch.optim.Adam([
        {'params': net.parameters(), 'lr': params["learning_rate"]},
        {'params': cell_mlp.parameters(), 'lr': params["mlp_learning_rate"]}
    ])

    # GAE model
    model = Model(num_drug_features, 256, 128).to(device)
    opt_model = torch.optim.Adam(model.parameters(),lr=params["gae_learning_rate"])
    # --------------------------------------------------------------------
    # Train. Iterate over epochs.
    # --------------------------------------------------------------------
    # Initialize early stopping parameters
    best_val_loss = float("inf")  # Keep track of the best validation loss
    patience = params["patience"]  # Number of epochs to wait before stopping
    patience_counter = 0  # Counter to track epochs without improvement
    # For tracking best models
    best_cnn_model, best_mlp_model, best_gae_model = None, None, None
    
    # Loop for training GAE
    for epoch_gae in range(params["gae_epochs"]):

        # Set GAE model to training mode
        model.train()
        
        # Forward pass through GAE model
        pred_gcn, gcn_feature = model(adj.to(device), node_feature.to(device))
        
        # Compute loss of GAE
        pred_1 = torch.mul(labels.to(device), pred_gcn)
        loss_model = norm * torch.mean(F.binary_cross_entropy_with_logits(input=pred_1.to(device), target=labels.float().to(device), pos_weight=pos_weight.to(device)))

        # Compute accuracy
        accuracy = acc(pred_1.to(device),labels.to(device))
        
        # Backpropagation for GAE
        opt_model.zero_grad()
        loss_model.requires_grad_(True)
        loss_model.backward()
        opt_model.step()
        print("GAE Epoch:", epoch_gae, "| Loss:", loss_model.item(), "| Accuracy:", accuracy)
        
        # -------------------------------------
        # Generate CNN-compatible input data
        # -------------------------------------
         # Detach gradients from GAE output
        x_matrix = gcn_feature
        x_matrix = x_matrix.detach().cpu().numpy()
        
        # Process cell feature matrix using MLP
        cell_out_train = cell_mlp(train_cell_features.to(device), 'sigmoid')
        cell_out_train = cell_out_train.detach().cpu().numpy()
        cell_out_val = cell_mlp(val_cell_features.to(device), 'sigmoid')
        cell_out_val = cell_out_val.detach().cpu().numpy()
        
        train_indices = train_response["split_index"].values  # Train indices
        val_indices = val_response["split_index"].values  # Validation indices
        
        drug1_train = drug1[train_indices]
        drug2_train = drug2[train_indices]
        drug_label_train = drug_label[train_indices]

        drug1_val = drug1[val_indices]
        drug2_val = drug2[val_indices]
        drug_label_val = drug_label[val_indices]
                # Combine GAE and MLP outputs
        x_new_matrix_train = new_matrix_with_cell(drug1_train, 
                                                  drug2_train, 
                                                  cell_out_train, 
                                                  x_matrix)
        x_new_matrix_val = new_matrix_with_cell(drug1_val, 
                                                drug2_val, 
                                                cell_out_val, 
                                                x_matrix)
        
        # Convert to PyTorch tensor and reshape for CNN
        x_new_matrix_train = torch.tensor(x_new_matrix_train)
        x_new_matrix_train = torch.reshape(x_new_matrix_train, (-1, 128, 1, 1))
        x_new_matrix_val = torch.tensor(x_new_matrix_val)
        x_new_matrix_val = torch.reshape(x_new_matrix_val, (-1, 128, 1, 1))
        
        # Convert drug labels to tensor
        train_drug_label_tensor = torch.tensor(drug_label_train).reshape(-1, 1)
        val_drug_label_tensor = torch.tensor(drug_label_val).reshape(-1, 1)
        
        # Create DataLoaders for train and val
        batch_size = params["batch_size"]
        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(x_new_matrix_train, train_drug_label_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = Data.DataLoader(
            dataset=Data.TensorDataset(x_new_matrix_val, val_drug_label_tensor),
            batch_size=batch_size,
            shuffle=False  # No shuffling for validation
        )
        
        # -------------------------------------
        # Train CNN and MLP Models
        # -------------------------------------
        for epoch_cnn in range(params["epochs"]):
            # Set CNN model to training mode
            net.train()
            # Set MLP model to training mode
            cell_mlp.train()
                
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
                
                # Forward pass through CNN
                pred_cnn = net(batch_x)
                
                # Compute CNN loss
                loss_net = torch.mean(F.binary_cross_entropy(pred_cnn, batch_y))

                # Compute training accuracy for CNN
                acc_train = acc(pred_cnn, batch_y)

                # Backpropagation for CNN
                opt_cnn.zero_grad()
                loss_net.backward()
                opt_cnn.step()

            # -------------------------------------
            # Validation Step
            # -------------------------------------
            net.eval()  # Set CNN to evaluation mode
            cell_mlp.eval()  # Set MLP to evaluation mode

            val_losses, val_accs = [], []

            with torch.no_grad():  # Disable gradients for validation
                for val_x_batch, val_y_batch in val_loader:
                    val_x_batch, val_y_batch = val_x_batch.to(device, dtype=torch.float32), val_y_batch.to(device, dtype=torch.float32)

                    # Forward pass through CNN on validation set
                    pred_val_cnn = net(val_x_batch)

                    # Compute validation loss for CNN
                    loss_val_cnn = torch.mean(F.binary_cross_entropy(pred_val_cnn, val_y_batch))
                    
                    # Compute validation accuracy for CNN
                    acc_val_cnn = acc(pred_val_cnn, val_y_batch)
                    
                    val_losses.append(loss_val_cnn.item())
                    val_accs.append(acc_val_cnn)
                    
            # Compute average validation loss and accuracy for CNN
            avg_val_loss_cnn = np.mean(val_losses)
            avg_val_acc_cnn = np.mean(val_accs)

            print(
            f"Epoch_GAE: {epoch_gae} | Epoch_CNN: {epoch_cnn} | Train Loss CNN: {loss_net.item()} | Train Acc CNN: {acc_train} | "
            f"Val Loss CNN: {avg_val_loss_cnn} | Val Acc CNN: {avg_val_acc_cnn} | "
            )
            
            # -------------------------------------
            # Early Stopping Logic
            # -------------------------------------
            if avg_val_loss_cnn < best_val_loss:
                best_val_loss = avg_val_loss_cnn
                best_cnn_model = net.state_dict()
                best_mlp_model = cell_mlp.state_dict()
                best_gae_model = model.state_dict()
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
                
        # Reset patience counter for early stopping after each GAE epoch
        patience_counter = 0
        best_val_loss = float("inf")  # Reset best validation loss
                  
    # Save the best models before exiting
    if "best_cnn_model" in locals():
        torch.save(best_cnn_model, modelpath)
        torch.save(best_mlp_model, mlp_modelpath)
        torch.save(best_gae_model, gae_modelpath)
        print("Best models have been saved.")
    else:
        print("No early stopping occurred; saving the last trained models.")
        torch.save(net.state_dict(), modelpath)
        torch.save(cell_mlp.state_dict(), mlp_modelpath)
        torch.save(model.state_dict(), gae_modelpath)


    print("All models have been saved successfully!")
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------
    net.load_state_dict(torch.load(modelpath, map_location=device))
    cell_mlp.load_state_dict(torch.load(mlp_modelpath, map_location=device))
    model.load_state_dict(torch.load(gae_modelpath, map_location=device))
    
    # Set models to evaluation mode
    net.eval()
    cell_mlp.eval()
    model.eval()
    
    # Extract features from MLP (cell embeddings)
    with torch.no_grad():
        val_cell_embeddings = cell_mlp(val_cell_features.to(device), "sigmoid")
        
    # Extract features from GAE (drug embeddings)
    with torch.no_grad():
        _, val_drug_embeddings = model(adj.to(device), node_feature.to(device))

    # Convert embeddings to numpy
    val_cell_embeddings = val_cell_embeddings.cpu().numpy()
    val_drug_embeddings = val_drug_embeddings.cpu().numpy()

    # Fuse MLP (cell) and GAE (graph) features
    val_fused_features = new_matrix_with_cell(drug1_val, drug2_val, val_cell_embeddings, val_drug_embeddings)

    # Convert back to Tensor and reshape for CNN input
    val_fused_features = torch.tensor(val_fused_features, dtype=torch.float32).to(device)
    val_fused_features = val_fused_features.reshape(-1, 128, 1, 1)  # Adjust based on CNN input shape

    # Get CNN predictions
    with torch.no_grad():
        y_pred_cnn = net(val_fused_features)

    # Move to CPU for evaluation
    y_pred = y_pred_cnn.cpu().numpy().flatten()
    y_pred_binary = (y_pred >= params["positive_threshold"]).astype(int)
    y_true = val_drug_label_tensor.cpu().numpy().flatten()

    # --------------------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # --------------------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_true,
        y_pred=y_pred_binary,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # --------------------------------------------------------------------
    # [Req] Compute performance scores
    # --------------------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=y_true,
        y_pred=y_pred_binary,
        y_prob=y_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="gaecds_original_params.txt",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])