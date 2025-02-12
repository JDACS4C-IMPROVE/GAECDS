import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from model_params_def import preprocess_params
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# [MODEL] Model-specific imports, as needed
import os
import numpy as np
import pandas as pd
import h5py

# [Req]
filepath = Path(__file__).resolve().parent

def assign_label(df, score_column="scores"):
    if score_column in df.columns:
        df["label"] = (df[score_column] > 0).astype(int)
    else:
        print(f"Column '{score_column}' not found in dataframe.")
    return df

# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the ML data files.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val/test sets
    # --------------------------------------------------------------------
    data_train_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    data_val_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")
    data_test_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    
    # --------------------------------------------------------------------
    # [Req] Create dataloaders and get response data - DRP specific
    # --------------------------------------------------------------------
    omics_obj = omics_utils.OmicsLoader(params)
    drugs_obj = drugs_utils.DrugsLoader(params)
    
    # Load response dataset per split
    response_train = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    response_val = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    response_test = drp.DrugResponseLoader(params,
                                    split_file=params["test_split_file"],
                                    verbose=False).dfs["response.tsv"]
    
    # Assign split indices (preserve original order)
    response_train["split_index"] = response_train.index
    response_val["split_index"] = response_val.index
    response_test["split_index"] = response_test.index
    
    # List of response datasets
    response_datasets = [response_train, response_val, response_test]
    
    # Loop through response datasets and get classification labels
    for i, res_df in enumerate(response_datasets):
        response_datasets[i] = assign_label(res_df, score_column="loewe")
        
    # Load and process full response dataset for GAE model
    response_all = pd.read_csv(os.path.join(params["input_dir"], "y_data/response.tsv"), sep="\t")
    response_all = assign_label(response_all, score_column="loewe")
    response_all = response_all[[params["drug_col_name_1"], params["drug_col_name_2"], params["canc_col_name"], "label"]]
        
    # --------------------------------------------------------------------
    # [Req] Load X data (feature representations)
    # --------------------------------------------------------------------
    drug_feat_df = drugs_obj.dfs["drug_ecfp4_nbits300.tsv"].reset_index()
    cell_feat_df = omics_obj.dfs["cancer_gene_expression.tsv"]
    
    # TODO Add check for IDs between response dataframe and features dataframes
    
    # --------------------------------------------------------------------
    # [MODEL] Preprocess X data
    # --------------------------------------------------------------------

    # TODO Create integer IDs to map to drug IDs
    
    # Drop the 'improve_chem_id' column
    drug_feat_df = drug_feat_df.drop(columns=[params["drug_col_name"]])
    
    # --------------------------------------------------------------------
    # [MODEL] Save X data
    # --------------------------------------------------------------------      

    # Define dictionary of stage-specific filenames
    stage_filenames = {
        "train": data_train_fname,
        "val": data_val_fname,
        "test": data_test_fname
    }
    
    for i, stage in enumerate(["train", "val", "test"]):
        print(f"Processing {stage} dataset...")

        # Get the corresponding response dataset
        res_df = response_datasets[i]
        
        print(f"Response data shape for {stage}: {res_df.shape}")

        # Filter `cell_feat_df` to match only cells in the current response dataset
        filtered_cell_df = cell_feat_df[cell_feat_df[params["canc_col_name"]].isin(res_df[params["canc_col_name"]])]

        # Ensure the order matches the response dataset
        filtered_cell_df = filtered_cell_df.set_index(params["canc_col_name"]).loc[res_df[params["canc_col_name"]]].reset_index()

        print(f"Cell data shape for {stage}: {filtered_cell_df.shape}")

        # Save both dataframes to HDF5 with separate datasets
        file_path = os.path.join(params["output_dir"], stage_filenames[stage])
        
        with pd.HDFStore(file_path, mode="w", complevel=9, complib="blosc") as store:
            store.put("response", res_df, format="table")
            store.put("cell_features", filtered_cell_df, format="table")

        print(f"Saved {stage_filenames[stage]} successfully with response and cell features.")
        
    # Save full response dataset
    response_all.to_csv(os.path.join(params["output_dir"], params["response_labels_file"]), index=False)
    
    # Save full fingerprints dataset
    drug_feat_df.to_csv(os.path.join(params["output_dir"], params["drug_fingerprints_file"]), index=False)
    
    print(f"Saved full response and drug features for GAE model.")

    # --------------------------------------------------------------------
    # [Req] Save response data (Y data)
    # --------------------------------------------------------------------  
    
    frm.save_stage_ydf(ydf=response_train, stage="train", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=response_val, stage="val", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=response_test, stage="test", output_dir=params["output_dir"])

    return params["output_dir"]

# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="gaecds_original_params.txt",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])