import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='RecVAE for Sequential Recommendation')

    parser.add_argument('--data_dir', type=str, default='./data/train/', help='Movielens train dataset path')
    parser.add_argument('--min_items_per_user', type=int, default=5)
    parser.add_argument('--min_users_per_item', type=int, default=0)
    parser.add_argument('--heldout_users', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--cuda', action='store_true', help='use CUDA')

    parser.add_argument('--hidden_dim', type=int, default=600)
    parser.add_argument('--latent_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='None') 
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--n_enc_epochs', type=int, default=3)
    parser.add_argument('--n_dec_epochs', type=int, default=1)
    parser.add_argument('--not_alternating', type=bool, default=False)
    parser.add_argument('--save', type=str, default='model/model.pt',
                        help='path to save the final model')
        
    args = parser.parse_args()

    return args