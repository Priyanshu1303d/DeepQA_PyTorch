from src.DeepQA.logging import logger
from src.DeepQA.pipeline.Stage_01_Data_Ingestion import DataIngestionPipeline
from src.DeepQA.pipeline.Stage_02_Data_Validation import DataValidationPipeline
from src.DeepQA.pipeline.Stage_03_Data_Transformation import DataTransformationPipeline
from src.DeepQA.pipeline.Stage_04_Model_Trainer import ModelTrainerPipeline
from src.DeepQA.pipeline.Stage_05_Model_Evaluation import ModelEvaluationPipeline
from src.DeepQA.pipeline.Stage_06_Model_Prediction import ModelPredictionPipeline


STAGE_NAME1 = "Data Ingetsion"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME1} started ---------------")
        training = DataIngestionPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME1} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME2 = "Data Validation"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME2} started ---------------")
        training = DataValidationPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME2} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME3 = "Data Transformation"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME3} started ---------------")
        training = DataTransformationPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME3} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME4 = "Model Training"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME4} started ---------------")
        training = ModelTrainerPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME4} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME5 = "Model Evaluation"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME5} started ---------------")
        training = ModelEvaluationPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME5} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME6 = "Model Prediction"

if __name__ == "__main__":
    try:
        logger.info(f"-------------------stage {STAGE_NAME6} started ---------------")
        training = ModelPredictionPipeline()
        training.main()
        logger.info(f"-----------stage {STAGE_NAME6} completed successfully ---------<<\n\n----")
    except Exception as e:
        logger.exception(e)
        raise e