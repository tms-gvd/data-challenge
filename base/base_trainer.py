import torch
from abc import abstractmethod
from numpy import inf
import wandb
import os

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            if not self.mnt_metric in ['train_loss', 'val_loss', 'val_iou']:
                raise ValueError("Unsupported metric: {} for monitoring".format(self.mnt_metric))
            else:
                print("Monitoring metric: {}".format(self.monitor))

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.save_period = cfg_trainer.get('save_period', self.epochs//10)

        if config.resume is not None:
            raise NotImplementedError("Resume is not implemented yet")
            # self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss, val_metric = self._train_epoch(epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                to_compare = val_metric if "val" in self.mnt_metric else train_loss
                
                if (to_compare < self.mnt_best and self.mnt_mode == "min") or (to_compare > self.mnt_best and self.mnt_mode == "max"):
                    print(f"New best found! MODE: {self.mnt_mode} | PREVIOUS BEST: {self.mnt_best:.3f} | NEW BEST: {to_compare}")
                    self.mnt_best = val_metric
                    not_improved_count = 0
                    self._save_checkpoint(epoch, save_best=True)
                else:
                    not_improved_count += 1

                # check early stopping condition
                if not_improved_count > self.early_stop:
                    print("Early stopping at epoch: ", epoch)
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)
            
            print()
        
        self._save_checkpoint(epoch, save_best=False)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best and self.config.with_wandb:
            print("Saving best checkpoint")
            path_save = str(self.config.save_dir / 'best_ckpt.pth')
            torch.save(state, path_save)
            wandb.save(path_save)
        elif not save_best and self.config.with_wandb:
            print("Saving last checkpoint")
            path_save = str(self.config.save_dir / 'last_ckpt.pth')
            torch.save(state, path_save)
            wandb.save(path_save)
        else:
            pass


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        raise NotImplementedError
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
