from src.DeepQA.logging import logger
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_01_Data_Ingestion import DataIngestion

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e