from src.DeepQA.logging import logger
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_05_Model_Evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(get_model_evaluation_config)
        
        accuracy , f1 = model_evaluation.evaluate_model()
        print(f1 ," " ,accuracy)
        model_evaluation.save_metrics(accuracy, f1)

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e