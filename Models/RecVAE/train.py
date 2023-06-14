import os
import wandb
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta

from src.utils import set_seeds, get_logger, create_directory, logging_conf
from src.args import parse_args
from src.wandb import wandb_settings
from src.dataloader import Preprocess, Dataloader
from src.model import RecVAE
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src.trainer import run, train, evaluate, inference
from src.metric import ndcg, recall


logger = get_logger(logger_conf=logging_conf)



def main(args):
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("🐰 TEAM NewRecs")
    
    # Weights & Biases Settings
    logger.info("1. Weights & Biases Settings ...")
    wandb.login()
    wandb_config, model_name, author, project, entity = wandb_settings(args)
    wandb.init(project=project, entity=entity, config=wandb_config)
    
    now = datetime.now(timezone(timedelta(hours=9)))
    wandb.run.name = f"{model_name}_{author}_{now.strftime('%m/%d %H:%M')}"
    
    # Data Preprocessing
    logger.info("2. Data Preprocessing ...")
    raw_data, unique_uid, tr_users, vd_users, te_users, n_users, unique_uid_before_shuffling = Preprocess(args).load_data_from_file(args)
    unique_sid, show2id, profile2id, id2show, id2profile = Preprocess(args).data_split(args, raw_data, unique_uid, tr_users, vd_users, te_users, unique_uid_before_shuffling)
    
    # Data Loading
    logger.info("3. Data Loading ...")
    loader = Dataloader(args.data_dir)
    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')
    
    
    # Build Model
    logger.info("4. Model Buliding ...")
    metrics = [{'metric': ndcg, 'k': 100}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]
    
    best_ndcg = -np.inf
    train_scores, valid_scores = [], []
    
    model = RecVAE(
        hidden_dim=args.hidden_dim, 
        latent_dim=args.latent_dim, 
        input_dim=train_data.shape[1]
        ).to(args.device)
    model_best = RecVAE(
        hidden_dim=args.hidden_dim, 
        latent_dim=args.latent_dim, 
        input_dim=train_data.shape[1]
        ).to(args.device)
    
    # Training
    logger.info("5. Training ...")
    
    learning_kwargs = {
        'model': model,
        'train_data': train_data,
        'batch_size': args.batch_size,
        'beta': args.beta,
        'gamma': args.gamma
    }
    
    encoder_params = set(model.encoder.parameters())
    decoder_params = set(model.decoder.parameters())
    
    optimizer_encoder = get_optimizer(args, params=encoder_params, model=model)
    optimizer_decoder = get_optimizer(args, params=decoder_params, model=model)
    
    scheduler_encoder = get_scheduler(optimizer_encoder)
    scheduler_decoder = get_scheduler(optimizer_decoder)
    
    
    run(args, model, model_best, optimizer_encoder, 
        optimizer_decoder, scheduler_encoder, scheduler_decoder, 
        learning_kwargs, metrics, ndcg, recall, train_scores, valid_scores, 
        train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, best_ndcg)

    
    # Create Submission File
    logger.info("6. Creating Submission File ...")
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    
    loader = Dataloader(args.data_dir)

    # load inference data
    inference_data = loader.load_data('inference')

    n_items = loader.load_n_items() # train_data.shape[1]

    logger.info(f'# of items: {n_items}')
    logger.info(f'# of users: {n_users}')
    
    inference_output = inference(args, model, inference_data)

    submission = []
    
    for idx in range(n_users):
        # item descending order
        sid_idx_preds_per_user = np.argsort(inference_output[idx])[::-1]
        sid_preds = unique_sid[sid_idx_preds_per_user]
        
        tmp = []
        for item in sid_preds:
            if len(tmp) == 10:
                break
            tmp.append((unique_uid_before_shuffling[idx], item))
        
        submission.extend(tmp)
        
    submission_df = pd.DataFrame(submission, columns=['user', 'item'])
    create_directory('./submit')
    KST = timezone(timedelta(hours=9))
    record_time = datetime.now(KST)
    write_path = os.path.join(f"./submit/RecVAE_submission_{record_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    submission_df.to_csv(write_path, index=False)
    logger.info("💫 Complete!")
if __name__ == "__main__":
    args = parse_args()
    main(args)