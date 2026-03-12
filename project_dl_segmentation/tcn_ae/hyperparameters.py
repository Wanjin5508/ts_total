from pathlib import Path 

hyperparameters = dict(
    version = "00",
    
    train_data_path = Path(r"..") ,
    test_data_path = Path(r".."),
    
    n_epochs = 100,
    batch_size = 32,
    
    dataloader = dict(
        n_samples = 10000,
        window_size = 200,
        window_shift = 200,
    ),
    
    lr = 0.001,
    
    # model config
    model = dict(
        n_inputs = 1,
        tcn_outputs = 32,
        n_feature_maps = 16,
        n_conv_layers = 8,
        kernel_size = 8,
        drop_out = 0.1,
        enc_dim = 4,
        enc_pool_size = 4
    )
)


