import argparse
import os
import sys
from logging import getLogger

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.general_recommender import BPR
from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import get_model, init_logger, init_seed
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from tqdm import tqdm
from datetime import datetime, timezone, timedelta

from utils import *

# set record_time
KST = timezone(timedelta(hours=9))
record_time = datetime.now(KST)

model_types = ["sequential", "general", "context_aware", "knowledge_aware", "exlib"]

if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="movie", help="name of datasets")
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument("--type", type=str, default="general", help="choose model type: 'general', 'context_aware', 'sequential', 'knowledge_aware', 'social'")
    
    args = parser.parse_args()
    args.config_files = os.path.join("./config", args.config_files)

    # choose file to inference
    # default: latest file
    BASE_DIR = "./model"
    FILE = ""
    file_list = os.listdir(BASE_DIR)
    print(file_list)

    if len(file_list) == 0:
        sys.exit(f"There is no .pth file in {BASE_DIR}")
    else:
        # TODO: pth 파일 이름 바꿔서 6-12-2023 이런식으로 변경하기 
        sorted_file_list = sorted(file_list, reverse=True, key=lambda x: x.split("-")[1:]) # 다 6월이니까 일단은..
        recent_file = sorted_file_list[0]
        FILE = recent_file
        print(FILE)

    # load model
    model_name = FILE.split("-")[0]
    model_type = ""

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join(BASE_DIR, FILE)
    )

    imported_model = get_model(model_name) # return model_class
    process = str(imported_model).split(".")

    for name in process:
        if "recommender" in name:
            model_type = name
            break
    else:
        print("Model name is not found.")

    config = Config(
        model=model_name, dataset=args.dataset, config_file_list=[args.config_files]
    )

    # load dataset
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    if model_type in model_types[0]: # Sequential Model
        prediction = generate_predict_seq(dataset, test_data, model, config)
    else: # General, Context_aware, Knowledge_aware, Exlib Model
        prediction = generate_predict(dataset, test_data, model, config)

    output_dir =  f"./submission/{model_name}_{record_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    submission = gererate_submission_from_prediction(prediction=prediction)
    submission.to_csv(output_dir, index=False)
    # gererate_submission_from_prediction(prediction=prediction)