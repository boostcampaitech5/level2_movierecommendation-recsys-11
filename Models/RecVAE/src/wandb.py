def wandb_settings(args):
    wandb_config = {
        "hidden_dim" : args.hidden_dim,
        "latent_dim" : args.latent_dim,
        "batch_size" : args.batch_size,
        "beta" : args.beta,
        "gamma" : args.gamma,
        "lr" : args.lr,
        "weight_decay" : args.weight_decay,
        "scheduler" : args.scheduler,
        "dropout_rate" : args.dropout_rate,
        "n_epochs" : args.n_epochs,
        "n_enc_epochs" : args.n_enc_epochs,
        "n_dec_epochs" : args.n_dec_epochs,
        "not_alternating" : args.not_alternating,
        "save" : args.save,
    }
    
    model_name = 'RecVAE'
    author = 'bles'
    project = 'movierec'
    entity = 'new-recs'
    
    return wandb_config, model_name, author, project, entity