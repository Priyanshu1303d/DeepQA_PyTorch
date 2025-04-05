from src.DeepQA.logging import logger
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_04_Model_Trainer import ModelTrainer
from torch.utils.data import DataLoader

STAGE_NAME = "Model Training stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_model_trainig_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(get_model_trainig_config)
        dataset = model_trainer.create_dataset()
        train_loader = DataLoader(dataset , batch_size= 1, shuffle=True,  pin_memory=True)
        model_trainer.train(train_loader)
        model_trainer.evaluate(train_loader)

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx==============x")

    except Exception as e:
        logger.exception(e)
        raise e