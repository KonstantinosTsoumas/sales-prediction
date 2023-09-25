import logging
import os
from datetime import datetime

timestamped_log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", timestamped_log_filename)
os.makedirs(logs_path, exist_ok = True)

log_file_path = os.path.join(logs_path, timestamped_log_filename)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] Line Number: %(lineno)d - %(name)s - Log Level: %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ =="__main__":
    logging.info("Logger started successfully")