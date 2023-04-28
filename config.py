def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.1}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.1, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg':  1.e-4,
                      'contrastive_reg':  1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))
    
    model_config = {'name': 'DOSE_drop3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000, 'aug_rate':0.5}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 100000}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_yelp_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    yelp_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512,'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000, 'aug_rate':0.5}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg':  1.e-4,
                      'contrastive_reg':  1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 800000}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_drop3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 1000000, 'aug_rate':0.7}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 300000}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    yelp_config.append((dataset_config, model_config, trainer_config))

    return yelp_config


def get_amazon_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Amazon/time',
                      'device': device}
    amazon_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0., 'feature_ratio': 1}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 10, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.9}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    amazon_config.append((dataset_config, model_config, trainer_config))
    
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 1000000}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    amazon_config.append((dataset_config, model_config, trainer_config))
    
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.6, 'aug_num': 1000000, 'aug_rate':0.7}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    amazon_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 1000000}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    amazon_config.append((dataset_config, model_config, trainer_config))
    return amazon_config


def get_alibaba_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/alibaba/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.1}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.1, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                      'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_drop3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000, 'aug_rate': 0.5}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 100000}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_ml_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/ml-1m/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer', 'device': device, 'n_epochs': 0,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.1}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMF', 'embedding_size': 64, 'n_layers': 0, 'device': device,
                    'dropout': 0.1, 'feature_ratio': 1.}
    trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'aux_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IMCGAE', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'IDCF_LGCN', 'embedding_size': 64, 'n_layers': 3, 'n_headers': 4,
                    'lgcn_path': 'lgcn.pth', 'device': device}
    trainer_config = {'name': 'IDCFTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'contrastive_reg': 1.e-3, 'device': device, 'n_epochs': 1000, 'batch_size': 2048,
                      'dataloader_num_workers': 6, 'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                      'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_drop3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 500000, 'aug_rate': 0.5}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'DOSE_aug_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 100000}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0.,
                      'contrastive_reg': 1.e-1,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512,
                      'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config



