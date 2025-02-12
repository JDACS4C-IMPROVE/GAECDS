import numpy as np
import pandas as pd
import os
import random

# --------------------------------------------------------------------
# Response Data
# --------------------------------------------------------------------
# Read in response data
res_df = pd.read_excel("./data/data_5693/data_all.xlsx")

# Rename relevant columns
res_df = res_df.rename(columns={
    "g_id1": "improve_chem_id_1",
    "g_id2": "improve_chem_id_2",
    "cell": "improve_sample_id",
    "scores": "loewe"
})

# Add source and study columns
res_df["source"] = "DrugComb"
res_df["study"] = "fake_exp"

# Reorder columns to match the target format
res_df = res_df[["source", "improve_sample_id", "improve_chem_id_1", "improve_chem_id_2", "study", "loewe"]]

# Ensure the y_data directory exists
y_dir = "improve_original/y_data"
os.makedirs(y_dir, exist_ok=True)

# Save as TSV
res_df.to_csv(os.path.join(y_dir, "response.tsv"), sep="\t", index=False)

print("Saved to y_data/: response.tsv")

# --------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------
# Determine dataset sizes
length = res_df.shape[0]
train_size = int(0.8 * length)
val_size = int(0.1 * length)
test_size = length - train_size - val_size  # Ensures full data utilization

# Shuffle and split the dataset
random_num = random.sample(range(0, length), length)

# Assign indices to train, validation, and test splits
train_num = random_num[:train_size]
val_num = random_num[train_size:train_size + val_size]
test_num = random_num[train_size + val_size:]

# Ensure the splits directory exists
splits_dir = "improve_original/splits"
os.makedirs(splits_dir, exist_ok=True)

# Save to text files in splits folder
with open(os.path.join(splits_dir, f"DrugComb_split_0_test.txt"), "w") as f:
    f.write("\n".join(map(str, test_num)))

with open(os.path.join(splits_dir, f"DrugComb_split_0_val.txt"), "w") as f:
    f.write("\n".join(map(str, val_num)))

with open(os.path.join(splits_dir, f"DrugComb_split_0_train.txt"), "w") as f:
    f.write("\n".join(map(str, train_num)))

print(f"Saved to splits/: DrugComb_split_0_test.txt, DrugCombo_split_0_val.txt, DrugCombo_split_0_train.txt")

# --------------------------------------------------------------------
# Drug Features
# --------------------------------------------------------------------
# Read in smiles file
smiles_df = pd.read_csv("./data/data_5693/smiles_197.csv")

smiles_df = smiles_df.rename(columns={
    "drug": "NAME",
    "g_id": "improve_chem_id",
    "smiles": "SMILES",
})

# Reorder columns to match the target format
smiles_df = smiles_df[["improve_chem_id", "NAME", "SMILES"]]

# Ensure the x_data directory exists
x_dir = "improve_original/x_data"
os.makedirs(x_dir, exist_ok=True)

# Save as TSV
smiles_df.to_csv(os.path.join(x_dir, "drug_info.tsv"), sep="\t", index=False)

print("Saved to x_data/: drug_info.tsv")

# Read in fingerprint data
fingerprint_df = pd.read_csv("./data/data_5693/feature197_300.csv")
# Concatenate to place it on the left
new_fingerprint_df = pd.concat([smiles_df[["improve_chem_id"]], fingerprint_df], axis=1)

# Save as TSV
new_fingerprint_df.to_csv(os.path.join(x_dir, "drug_ecfp4_nbits300.tsv"), sep="\t", index=False)

print("Saved to x_data/: drug_ecfp4_nbits300.tsv")

# --------------------------------------------------------------------
# Cell Features
# --------------------------------------------------------------------
# Read in gene expression data
cell_df = pd.read_csv("./data/data_5693/cell12.csv")

# Extract column headers (excluding the first column)
ensembl_ids = cell_df.columns[1:] 
num_gene_cols = len(ensembl_ids)
fake_entrez_ids = np.random.randint(5000, 70000, size=num_gene_cols)
fake_gene_names = [f"GENE{i}" for i in range(num_gene_cols)]

# Create MultiIndex Columns for gene expression data
multi_index_columns = pd.MultiIndex.from_arrays(
    [ensembl_ids,  # First level: Gene IDs
     fake_entrez_ids,  # Second level: Entrez IDs
     fake_gene_names],  # Third level: Gene Names
    names=["Gene ID", "Entrez ID", "Gene Name"]
)

# Reassign the column headers while keeping the first column unnamed
cell_df.columns = pd.MultiIndex.from_tuples([("", "", "")] + list(zip(ensembl_ids, fake_entrez_ids, fake_gene_names)))

# Save the merged DataFrame to a TSV file in the x_data folder
cell_df.to_csv(os.path.join(x_dir,'cancer_gene_expression.tsv'), sep='\t', index=False)

print(f"Saved to x_data/: cancer_gene_expression.tsv")