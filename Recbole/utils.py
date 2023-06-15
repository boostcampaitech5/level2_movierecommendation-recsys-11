# REF: https://github.com/RUCAIBox/RecBole/blob/a757b7c31bfa407c82847b4a06c8580fa1a6b45e/run_example/recbole-using-all-items-for-prediction.ipynb#L1246
# REF: https://recbole.io/docs/user_guide/usage/case_study.html
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_logger, init_seed
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from tqdm import tqdm


def generate_predict(dataset, 
                     test_data, 
                     model, 
                     config, 
                     user_data_file="./data/movie/unique_user.csv"
):
    # load unique user series
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    # prediction
    predicts = []

    for user in tqdm(users):
        # map external tokens to internal ids (str -> int)
        uid_series = dataset.token2id(          # returns: the internal ids (int or np.ndarray)
            dataset.uid_field,                  # field: field of external tokens (str)
            [user]                              # tokens: external tokens (str, list or np.ndarray)
        )

        # calculate the top-k items' scores and ids for each user in uid_series
        _, topk_iid_list = full_sort_topk(      # returns: topk_scores(torch.Tensor), topk_index(torch.Tensor)
            uid_series,                         # user id series (np.ndarray)
            model,                              # model to predict (AbstractRecommender)
            test_data,                          # the test_data of model (FullSortEvalDataLoader)
            k=10,                               # the top-k items (int)
            device=config["device"]             # device (torch.device, optional)
        )

        # map internal ids to external tokens (int -> str)
        external_item_list = dataset.id2token(  # returns: the external tokens (str or np.ndarray)
            dataset.iid_field,                  # field: field of internal ids (str)
            topk_iid_list.cpu()                 # ids: internal ids (int, list or np.ndarray)
        )

        # append prediction to token
        predicts.append(list(external_item_list[0]))

    return predicts


# process for sequence data 
def add_last_item(old_interaction,              # original interaction (recbole.dataset)
                  last_item_id,                 # last item id (int)
                  max_length=50):               # max sequence length (int)
    """
    Generate new sequence items 
    by adding last item to original interaction
    """

    # load last sequence item list from original interaction 
    new_seq_items = old_interaction["item_id_list"][-1]

    # last sequence length < max_len: add last item
    if old_interaction["item_length"][-1].item() < max_length:
        new_seq_items[old_interaction["item_length"][-1].item()] = last_item_id
    else: # last sequence length >= max_len: move items left
        new_seq_items = torch.roll(new_seq_items, -1)
        new_seq_items[-1] = last_item_id

    return new_seq_items.view(1, len(new_seq_items))


# prediction for sequence data
def predict_for_all_item(external_user_id, 
                         dataset, 
                         test_data, 
                         model, 
                         config
):
    """
    Predict using all items
    """
    model.eval()
    with torch.no_grad():
        uid_series = dataset.token2id(          # returns: the internal ids (int or np.ndarray)
            dataset.uid_field,                  # field: field of external tokens (str)
            [external_user_id]                  # tokens: external tokens (str, list or np.ndarray)
        )
        # load interaction of uid_series
        index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
        input_interaction = dataset[index]

        # set input for prediction (type: dict)
        inter_length = input_interaction['item_length'][-1].item()
        item_length = inter_length + 1 if model.max_seq_lenth > inter_length else model.max_seq_length
        
        test = {"item_id_list": add_last_item(input_interaction, inter_length, model.max_seq_length,),
                "item_length": torch.tensor([item_length]),}

        # prediction
        new_inter = Interaction(test)
        new_inter = new_inter.to(config["device"])
        scores = model.full_sort_predict(new_inter)
        scores = scores.view(-1, test_data.dataset.item_num) # (batch_size, item_num)
        scores[:, 0] = -np.inf # remove first item (first element in array is 'PAD' in Recbole)

    return torch.topk(scores, 10)


def generate_predict_seq(dataset, 
                         test_data, 
                         model, 
                         config, 
                         user_data_file="./data/movie/unique_user.csv"
):
    """
    Generate prediction from user profile data
    """
    # load unique user series
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    # prediction
    predict = []

    for user in tqdm(users):
        # predict
        temp = predict_for_all_item(user,       # user ids (str)
                                    dataset,    # dataset (recbole.dataset)
                                    test_data,
                                    model,
                                    config,
        )

        # map internal ids to external tokens (int -> str)
        external_item_list = dataset.id2token(  # returns: the external tokens (str or np.ndarray)
            dataset.iid_field,                  # field: field of internal ids (str)
            temp.indices.cpu(),                 # ids: internal ids (int, list or np.ndarray)
        )
        
        # append prediction to token
        predict.append(list(external_item_list[0]))

    return predict


def gererate_submission_from_prediction(prediction,
                                        user_data_file="./data/movie/unique_user.csv",
):
    """
    Generate submission file from prediction list
    """
    # load unique user series
    user_data = pd.read_csv(user_data_file, dtype=str)
    users = user_data["user"].unique()

    # prediction for submission
    result = []

    for index, user in enumerate(users):
        for item in prediction[index]:
            result.append([user, item])

    return pd.DataFrame(result, columns=["user", "item"])