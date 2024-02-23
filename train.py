import argparse
import collections
import torch
import numpy as np
import os

from parse_config import ConfigParser

import data_loader.datasets as module_datasets
import data_loader.data_loaders as module_dataloaders

import model.model as module_arch

import model.loss as module_loss
import model.metric as module_metric

from trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):

    # CREATE DATASET
    if 'dataset' in config.config.keys():
        print("Dataset:", config['dataset']['type'])
        print("Images labels path:", config['dataset']['args']['path_to_config'])
        dataset = config.init_obj('dataset', module_datasets)
    
        print("Data loader:", config['dataloader']['type'])
        data_loader = config.init_obj('dataloader', module_dataloaders, dataset)
        valid_data_loader = data_loader.split_validation()
    else:
        print("Data loader:", config['dataloader']['type'])
        data_loader = config.init_obj('dataloader', module_dataloaders)
        valid_data_loader = data_loader.split_validation()
    
    print()
    
    # CREATE MODEL
    print("Model:", config['arch']['type'])
    model = config.init_obj('arch', module_arch)
    config.config['num_classes'] = config.config['arch']['args']['num_classes']
    os.environ['NUM_CLASSES'] = str(config.config['num_classes'])

    # prepare for (multi-device) GPU training
    # device, device_ids = prepare_device(config['n_gpu'])
    device = config.config.get('device', 'cpu')
    device_ids = []
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # model to device
    model = model.to(device, dtype=torch.float64)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print()
    
    # CREATE OPTIMIZER, LOSS, METRICS
    if isinstance(config['loss'], dict):
        criterion = config.init_obj('loss', torch.nn)
        print("Loss:", config['loss']["type"])
    
    elif isinstance(config['loss'], str):
        if config['loss'] == "bce+iou":
            criterion = (torch.nn.BCELoss(), module_metric.IoU())
        else:
            criterion = getattr(module_loss, config['loss'])
        print("Loss:", config['loss'])
    
    else:
        raise ValueError("Loss must be a string or a dictionary")

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    print("Metrics:")
    for metric in metrics:
        print(f"\t - {metric.__name__}")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    print("Optimizer:", config['optimizer']["type"])
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    print("Learning rate scheduler:", config['lr_scheduler']["type"])
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    print()
    
    print("Starting training")
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str,
    #                   help='indices of GPUs to enable (default: all)')
    args.add_argument('--with_wandb', default=False, action='store_true',
                      help='Use wandb for logging')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--path_data'], type=str, target='dataset;args;path_to_config'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--device'], type=str, target='device'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
