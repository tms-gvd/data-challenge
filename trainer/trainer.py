import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.config = config
        self.device = device

        # train dataloader
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            raise NotImplementedError
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        self.step = 0

        # validation dataloader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        # learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # metrics functions, is a list of functions to call on (output, target)
        self.metrics = {metric.__name__: metric() for metric in metric_ftns}

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        running_loss = 0.0
        running_count = 0

        for data, target in tqdm(
            self.data_loader,
            desc=f"TRAIN {epoch}/{self.epochs}",
            ncols=100,
        ):
            # move data to device
            data, target = data.to(self.device), target.to(self.device)

            # back-prop step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_count += data.size(0)

            # log losses and metrics
            batch_metrics = {"train/loss": loss.item()}
            for name, metric in self.metrics.items():
                batch_metrics[f"train/{name}"] = metric(output, target)
            wandb.log(batch_metrics, step=self.step)
            
            self.step += 1

            # if iterations instead of epochs, break after len_epoch
            # if batch_idx == self.len_epoch:
            #     break

        train_loss = running_loss / running_count
        if not self.config.with_wandb:
            print(f"TRAIN loss: {train_loss}")
        
        # validation
        if self.do_validation:
            val_loss = self._valid_epoch(epoch)

        # learning rate scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return train_loss, val_loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        running_loss = 0.0
        running_count = 0
        
        running_metrics = {f'val/{name}': 0 for name in self.metrics}

        with torch.no_grad():
            for data, target in tqdm(
                self.valid_data_loader,
                desc=f"VAL {epoch}/{self.epochs}",
                ncols=100,
            ):
                # move data to device
                data, target = data.to(self.device), target.to(self.device)

                # predictions and loss
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                running_count += data.size(0)
                
                for name, metric in self.metrics.items():
                    running_metrics[f'val/{name}'] += metric(output, target)

            epoch_loss = running_loss / running_count
            
            # log losses and metrics
            running_metrics["val/loss"] = epoch_loss
            wandb.log(running_metrics, step=self.step)

            if not self.config.with_wandb:
                print(f"VAL loss: {epoch_loss}")

        return epoch_loss
