import os
import json

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建配置文件的绝对路径
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as jsonfile:
    config = json.load(jsonfile)


class LIDConfig(object):

    dir = config['dataDir']['dir']

    train_path = config['dataDir']['train_path'].replace('$dir', dir)

    test_path = config['dataDir']['test_path'].replace('$dir', dir)

    val_path = config['dataDir']['val_path'].replace('$dir', dir)

    batch_size = int(config['model_parameters']['batch_size'])
    epochs = int(config['model_parameters']['epochs'])

    model_type = config['model_parameters']['model_type']
    
    # upstream model to be loaded from s3prl. Some of the upstream models are: wav2vec2, hubert, TERA, mockingjay etc.
    #See the available models here: https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/README.md
    upstream_model = config['model_parameters']['upstream_model']


    # feature dimension of upstream model. For example, 
    # For wav2vec2, feature_dim = 768
    # For npc, feature_dim = 512
    # For tera, feature_dim = 768
    feature_dim = int(config['model_parameters']['feature_dim'])

    # No of GPUs for training and no of workers for datalaoders
    gpu = int(config['gpu'])
    n_workers = int(config['n_workers'])

    # model checkpoint to continue from
    model_checkpoint = config['dataDir']['model_checkpoint']
    results_path = config['dataDir']['results_path']
    
    # LR of optimizer
    lr = float(config['model_parameters']['lr'])
    mixup_type = config['model_parameters']['mixup_type']
    cluster = config['model_parameters']['cluster']
    # 是否解冻上游模型的最后几个卷积层
    unfreeze_last_conv_layers = bool(config['model_parameters'].get('unfreeze_last_conv_layers', False))

    run_name = config['run_name']