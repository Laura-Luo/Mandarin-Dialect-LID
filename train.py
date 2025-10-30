import os
from argparse import ArgumentParser

import torch
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import LIDConfig

# SEED
SEED=100
pl.seed_everything(SEED)
torch.manual_seed(SEED)

os.environ['WANDB_MODE'] = 'online'


from Datasets.datasetLID import LIDDataset, collate_fn
from Models.lightning import LightningModel
from Models.datamaps import DataMapsCallback, embed_datamaps_into_dataframe, generate_maps_plots_from_dataframe
import pandas as pd


class FirstEpochStoppingCallback(Callback):
    """
    自定义回调，在第一个epoch结束后正常终止训练
    """
    def on_train_epoch_end(self, trainer, pl_module):
        # 检查是否是第一个epoch结束
        if trainer.current_epoch == 0:
            print("\n第一个epoch训练完成，正在正常终止训练...")
            trainer.should_stop = True


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--train_path', type=str, default=LIDConfig.train_path)
    parser.add_argument('--val_path', type=str, default=LIDConfig.val_path)
    parser.add_argument('--test_path', type=str, default=LIDConfig.test_path)
    parser.add_argument('--batch_size', type=int, default=LIDConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=LIDConfig.epochs)
    parser.add_argument('--feature_dim', type=int, default=LIDConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=LIDConfig.lr)
    parser.add_argument('--gpu', type=int, default=LIDConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=LIDConfig.n_workers)
    parser.add_argument('--dev', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=LIDConfig.model_checkpoint)
    parser.add_argument('--model_type', type=str, default=LIDConfig.model_type)
    parser.add_argument('--upstream_model', type=str, default=LIDConfig.upstream_model)
    parser.add_argument('--mixup_type', type=str, default=LIDConfig.mixup_type)
    parser.add_argument('--cluster', type=str, default=LIDConfig.cluster)
    parser.add_argument('--unfreeze_last_conv_layers', action='store_true', default=LIDConfig.unfreeze_last_conv_layers)
    parser.add_argument('--noise_dataset_path', type=str, default=None)
    
    hparams = parser.parse_args()
    print(f'Training Model on LID Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = LIDDataset(
        CSVPath = hparams.train_path,
        hparams = hparams,
        is_train=True,
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )
    ## Validation Dataset
    valid_set = LIDDataset(
        CSVPath = hparams.val_path,
        hparams = hparams,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=hparams.batch_size,
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    print('Dataset Split (Train, Validation)=', len(train_set), len(valid_set))
    logger = WandbLogger(
        name=LIDConfig.run_name,
        project='LangID'
    )
    
    HPARAMS= vars(hparams)
    model = LightningModel(HPARAMS)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor='val/acc', 
        mode='max',
        verbose=1)

    # lr_monitor = LearningRateMonitor(logging_interval='step')
    
    datamaps_cb = DataMapsCallback(log_dir='/root/Langid/results/plots', dataset_length=len(train_set))
    # first_epoch_stopping = FirstEpochStoppingCallback()

    trainer = Trainer(
        fast_dev_run=hparams.dev, 
        devices=hparams.gpu,
        accelerator='gpu' if hparams.gpu > 0 else 'cpu',
        accumulate_grad_batches=4,
        max_epochs=hparams.epochs, 
        callbacks=[
            EarlyStopping(
                monitor='train/loss',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='min' 
                ),
            datamaps_cb,
            model_checkpoint_callback,
            #first_epoch_stopping,  # 添加第一个epoch停止回调
            # lr_monitor,
        ],
        logger=logger,
        )

    trainer.fit(model, trainloader, valloader)

    df = pd.read_csv(train_set.CSVPath).reset_index(drop=True)
    print(f"\nProcessing datamaps with {len(datamaps_cb.metrics)} collected metrics")
    # 使用datamaps_cb实例的metrics而不是全局metrics
    df = embed_datamaps_into_dataframe(df, datamaps_cb.metrics)
    print("Generating datamaps plots...")
    generate_maps_plots_from_dataframe(df, '/root/Langid/results/plots')
    print("Datamaps saved successfully!")