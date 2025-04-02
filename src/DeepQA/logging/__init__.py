import sys
import os
import logging


log_string =  "[%(asctime)s : %(levelname)s : %(module)s  : %(message)s]"


log_dir = "logs"
log_filepath = os.path.join(log_dir , "running_logs.log")
os.makedirs(log_dir , exist_ok=True)


logging.basicConfig(
    level = logging.INFO,
    format = log_string,

    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)  # Uses the module name automatically


