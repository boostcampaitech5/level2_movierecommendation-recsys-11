import torch
import numpy as np
import pandas as pd
import time
import wandb
from .utils import get_logger, logging_conf, generate
from copy import deepcopy
from .metric import ndcg, recall

logger = get_logger(logger_conf=logging_conf)


def run(args, model, model_best, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, learning_kwargs, metrics, ndcg, recall, train_scores, valid_scores, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, best_ndcg):
    for epoch in range(args.n_epochs):
        if args.not_alternating:
            train(args, opts=[optimizer_encoder, optimizer_decoder], sches=[scheduler_encoder, scheduler_decoder], epochs=1, dropout_rate=args.dropout_rate, **learning_kwargs)
        else:
            train(args, opts=[optimizer_encoder], sches=[scheduler_encoder], n_epochs=args.n_enc_epochs, dropout_rate=args.dropout_rate, **learning_kwargs)
            model.update_prior()
            train(args, opts=[optimizer_decoder], sches=[scheduler_decoder], n_epochs=args.n_dec_epochs, dropout_rate=0, **learning_kwargs)
        
        train_scores.append(
            evaluate(args, model, train_data, train_data, metrics, 0.01)[0]
        )
        
        valid_scores.append(
            evaluate(args, model, vad_data_tr, vad_data_te, metrics, 1)[0]
        )
        
        if valid_scores[-1] > best_ndcg:
            best_ndcg = valid_scores[-1]
            model_best.load_state_dict(deepcopy(model.state_dict()))
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            
        logger.info(f'🚀 | epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
                f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')
        
        if (epoch % 10) == 0:
            wandb.log({"best valid": best_ndcg, "valid ndcg@100": valid_scores[-1], "train ndcg@100": train_scores[-1]})

    # test data  
    test_metrics = [{'metric': ndcg, 'k': 100}, {'metric': recall, 'k': 20}, {'metric': recall, 'k': 50}]

    final_scores = evaluate(args, model_best, test_data_tr, test_data_te, test_metrics)

    for metric, score in zip(test_metrics, final_scores):
        logger.info(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")

    wandb.log({"test ndcg@100": final_scores[0], "test recall@20": final_scores[1], "test recall@50": final_scores[2]})



def train(args, model, opts, sches, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
    # training mode
    model.train()
  
    for epoch in range(n_epochs):
        for batch in generate(args, batch_size, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()
      
            for optimizer in opts:
                optimizer.zero_grad()
      
            _, loss = model(ratings, beta, gamma, dropout_rate)
            loss.backward()
      
            for optimizer in opts:
                optimizer.step()
        
        # learning rate scheduling
        for scheduler in sches:
            scheduler.step()

def evaluate(args, model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
    # evaluation mode
    model.eval()
    metrics = deepcopy(metrics)
  
    for m in metrics:
        m['score'] = []
    
    for batch in generate(args, 
                          batch_size, 
                        data_in, 
                        data_out,
                        samples_perc_per_epoch):
        
        items_in = batch.get_ratings_to_dev()
        items_out = batch.get_ratings(is_out=True)
        
        items_pred = model(items_in, calculate_loss=False).cpu().detach().numpy()
        
        if not(data_in is data_out):
            items_pred[batch.get_ratings().nonzero()] = -np.inf
        
        for m in metrics:
            m['score'].append(m['metric'](items_pred, items_out, k=m['k']))
        
    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
        
    return [x['score'] for x in metrics]




def inference(args, model, data_in, samples_perc_per_epoch=1, batch_size=500):
    model.eval()
    output = []
  
    with torch.no_grad():
        for batch in generate(args,
                            batch_size, 
                              data_in,
                              samples_perc_per_epoch):
      
            ratings_in = batch.get_ratings_to_dev()
      
            ratings_pred = model(ratings_in, calculate_loss=False).detach().cpu().numpy()
      
            # remove watched items
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
            
            n_users = ratings_pred.shape[0]
            for i in range(n_users):
                output.append(ratings_pred[i])
                
    return output