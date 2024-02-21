import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_dataloaders
import data_loader.datasets as module_datasets
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from preprocessing import Preprocessor, resize_image, gray_to_rgb
from preprocessing import timesteps_to_classes, timesteps_to_one_hot

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):

    # 2. Create dataset
    print("Dataset:", config['dataset']['type'])
    print("Images labels path:", config['dataset']['args']['images_labels_path'])
    dataset = config.init_obj('dataset', module_datasets)
    
    # 3. Create data_loader
    print("Data loader:", config['dataloader']['type'])
    data_loader = config.init_obj('dataloader', module_dataloaders, dataset)
    valid_data_loader = data_loader.split_validation()
    
    print()
    
    # 4. Create model
    print("Model:", config['arch']['type'])
    model = config.init_obj('arch', module_arch)

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
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    print()
    
    # 5. Create loss, metric
    print("Loss:", config['loss'])
    criterion = getattr(module_loss, config['loss'])
    print("Metrics:", config['metrics'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
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
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--device'], type=str, target='device'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
