
from DeepQA.logging import logger
from DeepQA.utils.common import read_yaml , create_directories , get_size
from DeepQA.constants import *
from DeepQA.entity.entity_config import (DataIngestionConfig , DataValidationConfig , DataTransformationConfig ,
                                          ModelTrainerConfig, ModelPredictionConfig , ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(self , config_filepath =  CONFIG_FILE_PATH , params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir,
        )
        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            ALL_REQUIRED_FILES= config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])
        data_transformation = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            vocab_file_path = config.vocab_file_path,
            output_dir= config.output_dir
        )

        return data_transformation
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer

        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path= config.data_path,
            output_path= config.output_path,
            vocab_file_path = config.vocab_file_path,
            epochs = params.epochs,
            weight_decay = params.weight_decay,
            learning_rate = params.learning_rate,
            optimizer = params.optimizer
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir= config.root_dir,
            model_metrics_json= config.model_metrics_json,
            saved_model_path= config.saved_model_path,
            vocab_file_path= config.vocab_file_path,
            data_path= config.data_path

        )

        return model_evaluation_config
    

    def get_model_prediction_config(self) -> ModelPredictionConfig:
        config = self.config.model_prediction

        model_prediction_config = ModelPredictionConfig(
            saved_model_path= config.saved_model_path,
            vocab_file_path = config.vocab_file_path
        )
        return model_prediction_config
