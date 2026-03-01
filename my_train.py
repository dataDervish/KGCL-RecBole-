from recbole.quick_start import run_recbole

# 你的自定义配置
config_dict = {
    'model': 'KGCL',
    'dataset': 'ml-1m',
    'device': 'cuda',
    'train_batch_size': 8192,
}

run_recbole(config_dict=config_dict)