from src.DeepQA.logging import logger
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_06_Model_Prediction import Predictor

STAGE_NAME = "Model Prediction stage"

class ModelPredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_model_prediction_config  = config.get_model_prediction_config()
        model_prediction = Predictor(get_model_prediction_config)
        answer = model_prediction.predict("Which ocean is the largest?")
        print(f"The answer is : ",answer)

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelPredictionPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e