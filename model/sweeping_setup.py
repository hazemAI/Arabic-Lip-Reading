import wandb

# -------------------------------------------------------------------------
#  Hyperparameter Sweep Configuration
# -------------------------------------------------------------------------
# Define the search strategy, evaluation metric, and parameter space
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val_loss',  # name of the metric to optimize
        'goal': 'minimize'
    },
    'parameters': {
        # Optimizer & scheduler
        'initial_lr': {'values': [1e-3, 3e-4]},
        'warmup_epochs': {'values': [0, 4, 8]},
        'total_epochs': {'values': [60, 80, 100]},

        # CTC/Attention loss weighting
        'ctc_weight': {'values': [0.0, 0.1, 0.2, 0.3]},
        'label_smoothing': {'values': [0.1, 0.2]},

        # Temporal encoder choice and config
        'encoder_type': {'values': ['densetcn', 'conformer', 'mstcn']},
        # Densetcn
        'densetcn_block_config': {'values': [[2,2,2,2], [3,3,3,3]]},
        'densetcn_growth_rate_set': {'values': [[96,96,96,96], [192,192,192,192]]},
        'densetcn_reduced_size': {'values': [256, 512, 768]},
        'densetcn_kernel_size_set': {'values': [[3, 5, 7], [5, 7, 9]]},
        'densetcn_dilation_size_set': {'values': [[1, 2, 4], [1, 2, 4, 8]]},
        'densetcn_hidden_dim': {'values': [256, 512, 768]},
        # MSTCN
        'mstcn_hidden_dim': {'values': [256, 512, 768]},
        'mstcn_num_channels': {'values': [[96,96,96,96], [192,192,192,192]]},
        # Conformer
        'conformer_attention_dim': {'values': [256, 512, 768]},
        'conformer_attention_heads': {'values': [4, 8]},
        'conformer_linear_units': {'values': [512, 1024, 2048]},
        'conformer_num_blocks': {'values': [4, 8, 12]},
        # Decoder
        'decoder_attention_dim': {'values': [256, 512, 768]},
        'decoder_attention_heads': {'values': [4, 8]},
        'decoder_linear_units': {'values': [512, 1024, 2048]},
        'decoder_num_blocks': {'values': [2, 4, 6]},
    }
}


def train():
    """
    This function is called by W&B for each hyperparameter combination.
    It imports and runs your existing training loop, which must use wandb.config values.
    """
    # Initialize a new run (your master script should use wandb.config to read these)
    wandb.init()

    # Import and execute your training entrypoint
    from master_monitoring import train_model
    train_model()


if __name__ == '__main__':
    # Create the sweep in your project (will return sweep_id)
    sweep_id = wandb.sweep(sweep_config, project='arabic-lipreading-avsr')
    # Launch sweep agent to run train() across parameter combinations
    wandb.agent(sweep_id, function=train, count=40) 