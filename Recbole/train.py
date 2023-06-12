import argparse
import os
from logging import getLogger
import wandb

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.trainer import Trainer
from recbole.utils import get_model, init_logger, init_seed

if __name__ == "__main__":

    # argument 입력하기
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="BPR", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="movie", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    args = parser.parse_args()

    # config file 입력하기
    args.config_files = os.path.join("./config", args.config_files)
    config = Config(
        model=args.model, dataset=args.dataset, config_file_list=[args.config_files]
    )
    
    wandb.init(project="movierec", entity="new-recs")
    
    wandb.run.name = args.model + "_MS"
    
    wandb.config.model = args.model
    wandb.config.dataset = args.dataset
    wandb.config.config_files = args.config_files

    # 랜덤 시드 고정하기
    init_seed(config["seed"], config["reproducibility"])

    # 로깅
    init_logger(config)
    logger = getLogger()

    # Config 파일 확인하기
    logger.info(config)

    # Custom Dataset 만들고 분할
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 모델 불러온 후 확인
    imported_model = get_model(args.model)
    model = imported_model(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # Trainer 설정
    trainer = Trainer(config, model)

    # 모델 학습
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    
    wandb.log({"best_valid_score": best_valid_score, "best_valid_result": best_valid_result})

    # 모델 평가
    test_result = trainer.evaluate(test_data)
    print(test_result)
    
    wandb.log({"test_result": test_result})