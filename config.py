# config.py

# Directories
ARTIFACTS_DIR = 'artifacts'
SRC_DIR = 'src'
COMPONENTS_DIR = f'{SRC_DIR}/components'
PIPELINE_DIR = f'{SRC_DIR}/pipeline'
LOGS_DIR = 'logs'
INPUT_DIR = 'input'
TESTS_DIR = 'tests'

# Artifacts Paths
ENCODED_DATA_CSV_PATH = f'{ARTIFACTS_DIR}/encoded_data.csv'
MODEL_PKL_PATH = f'{ARTIFACTS_DIR}/model.pkl'
PREPROCESSOR_PKL_PATH = f'{ARTIFACTS_DIR}/preprocessor.pkl'
ENCODER_OBJ_FILE_PATH = f'{ARTIFACTS_DIR}/encoder.pkl'

# Data Paths
INPUT_DATA_CSV = f'{INPUT_DIR}/DataCoSupplyChainDataset.csv'

# Log Paths
LOG_FILE_PATH = f'{LOGS_DIR}/app.log'