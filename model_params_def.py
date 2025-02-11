from improvelib.utils import str2bool

preprocess_params = [
    {
        "name": "drug_col_name_1",
        "type": str,
        "default": "improve_chem_id_1",
        "help": "Column name for the first drug identifier in the dataset."
    },
    {
        "name": "drug_col_name_2",
        "type": str,
        "default": "improve_chem_id_2",
        "help": "Column name for the second drug identifier in the dataset."
    },
    {
        "name": "response_labels_file",
        "type": str,
        "default": "response_all_label_only.csv",
        "help": "Path to the CSV file containing response labels for training the GAE model. Defaults to 'response_all_label_only.csv'.",
    },
    {
        "name": "drug_fingerprints_file",
        "type": str,
        "default": "drug_ecfp4_nbits300.csv",
        "help": "Path to the CSV file containing drug ECFP4 fingerprints (300-bit) for training the GAE model. Defaults to 'drug_ecfp4_nbits300.csv'.",
    },
]


train_params = [
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference (e.g., 'cuda:0', 'cuda:1'). Defaults to 'cuda:0'.",
    },
    {
        "name": "drug_col_name_1",
        "type": str,
        "default": "improve_chem_id_1",
        "help": "Column name for the first drug identifier in the dataset."
    },
    {
        "name": "drug_col_name_2",
        "type": str,
        "default": "improve_chem_id_2",
        "help": "Column name for the second drug identifier in the dataset."
    },
    {
        "name": "canc_col_name",
        "type": str,
        "default": "improve_sample_id",
        "help": "Column name for the cell line/sample identifier in the dataset."
    },
    {
        "name": "gae_model_file_name",
        "type": str,
        "default": "gae_model.pt",
        "help": "File name for saving the GAE model.",
    },
    {
        "name": "mlp_model_file_name",
        "type": str,
        "default": "mlp_model.pt",
        "help": "File name for saving the MLP model.",
    },
    {
        "name": "gae_epochs",
        "type": int,
        "default": 5,
        "help": "Number of epochs for training the GAE model.",
    },
    {
        "name": "mlp_learning_rate",
        "type": float,
        "default": 0.01,
        "help": "Learning rate for the MLP model.",
    },
    {
        "name": "gae_learning_rate",
        "type": float,
        "default": 0.00001,
        "help": "Learning rate for the GAE model.",
    },
    {
        "name": "response_labels_file",
        "type": str,
        "default": "response_all_label_only.csv",
        "help": "Path to the CSV file containing response labels for training the GAE model. Defaults to 'response_all_label_only.csv'.",
    },
    {
        "name": "drug_fingerprints_file",
        "type": str,
        "default": "drug_ecfp4_nbits300.csv",
        "help": "Path to the CSV file containing drug ECFP4 fingerprints (300-bit) for training the GAE model. Defaults to 'drug_ecfp4_nbits300.csv'.",
    },
    {
        "name": "positive_threshold",
        "type": float,
        "default": 0.5,
        "help": """Threshold for classifying a prediction as a positive (synergistic) interaction.
        Predictions with values greater than or equal to this threshold are labeled as synergistic (1), 
        while lower values are labeled as non-synergistic (0).""",
    },
]


infer_params = [
    {
        "name": "hidden",
        "type": int,
        "default": 8192,
        "help": "Hidden layer size for the neural network.",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference (e.g., 'cuda:0', 'cuda:1'). Defaults to 'cuda:0'.",
    },
    {
        "name": "drug_col_name_1",
        "type": str,
        "default": "improve_chem_id_1",
        "help": "Column name for the first drug identifier in the dataset."
    },
    {
        "name": "drug_col_name_2",
        "type": str,
        "default": "improve_chem_id_2",
        "help": "Column name for the second drug identifier in the dataset."
    },
    {
        "name": "canc_col_name",
        "type": str,
        "default": "improve_sample_id",
        "help": "Column name for the cell line/sample identifier in the dataset."
    },
        {
        "name": "gae_model_file_name",
        "type": str,
        "default": "gae_model.pt",
        "help": "File name for saving the GAE model.",
    },
    {
        "name": "mlp_model_file_name",
        "type": str,
        "default": "mlp_model.pt",
        "help": "File name for saving the MLP model.",
    },
    {
        "name": "gae_epochs",
        "type": int,
        "default": 5,
        "help": "Number of epochs for training the GAE model.",
    },
    {
        "name": "mlp_learning_rate",
        "type": float,
        "default": 0.01,
        "help": "Learning rate for the MLP model.",
    },
    {
        "name": "gae_learning_rate",
        "type": float,
        "default": 0.00001,
        "help": "Learning rate for the GAE model.",
    },
    {
        "name": "positive_threshold",
        "type": float,
        "default": 0.5,
        "help": """Threshold for classifying a prediction as a positive (synergistic) interaction.
        Predictions with values greater than or equal to this threshold are labeled as synergistic (1), 
        while lower values are labeled as non-synergistic (0).""",
    },
    {
        "name": "response_labels_file",
        "type": str,
        "default": "response_all_label_only.csv",
        "help": "Path to the CSV file containing response labels for training the GAE model. Defaults to 'response_all_label_only.csv'.",
    },
    {
        "name": "drug_fingerprints_file",
        "type": str,
        "default": "drug_ecfp4_nbits300.csv",
        "help": "Path to the CSV file containing drug ECFP4 fingerprints (300-bit) for training the GAE model. Defaults to 'drug_ecfp4_nbits300.csv'.",
    },
]