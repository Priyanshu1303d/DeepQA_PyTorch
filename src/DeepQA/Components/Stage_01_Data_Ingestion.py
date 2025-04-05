import os
import gdown
from DeepQA.logging import logger
from pathlib import Path
from DeepQA.config.configuration import DataIngestionConfig
from DeepQA.utils.common import *
import zipfile

class DataIngestion :
    def __init__(self , config = DataIngestionConfig):
        self.config = config


    def download_file(self):
        '''
            This method downloads the dataset from the Google drive        
        '''
        source_url = self.config.source_url
        unzip_path = self.config.unzip_dir

        if not os.path.exists(self.config.local_data_file):
            gdown.download_folder(url = source_url , output = unzip_path , use_cookies=False , quiet=  False)
            logger.info(f"Dataset downloaded!!")
        else:
            logger.info(f"file already exists of size : {get_size(Path(self.config.local_data_file))}")



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path , exist_ok= True)

        with zipfile.ZipFile(self.config.local_data_file , 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Extraction Completed")