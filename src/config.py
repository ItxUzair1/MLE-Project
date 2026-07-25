import yaml
import os  # noqa: E402

def load_yaml(path:str="config/config.yaml")-> dict:
    try:
        with open(path,"r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at :{path}")
        
config=load_yaml()

LOG_DIR=config["logs"]["log_dir"]
DATABASE_PATH=config["data"]["database_path"]
TABLE_NAME=config["data"]["table_name"]
RAW_PATH=config["data"]["raw_path"]
PROCESSED_DATA_PATH=config["data"]["processed_path"]
PREPROCESSOR_PATH=config["artifacts"]["preprocessor_path"]


TARGET_COLUMN="Churn"
TEST_SIZE=config["model"]["test_size"]
MODEL_PATH=config["artifacts"]["model_path"]
MODEL_NAME=config["model"]["model_name"]
EXPERIMENT_NAME=config["model"]["experiment_name"]
SHAP_BACKGROUND_PATH=config["artifacts"]["shap_background_path"]


DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "muhammaduzair1411")
DAGSHUB_REPO_NAME  = os.getenv("DAGSHUB_REPO_NAME",  "my-first-repo")