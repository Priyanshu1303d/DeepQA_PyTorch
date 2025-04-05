from src.DeepQA.logging import logger
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_03_Data_Transformation import DataTransformation

STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation()
        data_transformation = DataTransformation(data_transformation_config)
        
        df = data_transformation.load_dataset()
        data_transformation.build_vocab(df)
        df = data_transformation.df_to_indices(df)
        data_transformation.save_dataset(df)

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e