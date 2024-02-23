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
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device)

            # back-prop step
            self.optimizer.zero_grad()
            output = self.model(data)
            if isinstance(self.criterion, tuple):
                bce, iou = self.criterion
                y_preds = torch.sort(torch.topk(output, 4).indices, dim=1).values
                y_true = torch.sort(torch.topk(target, 4).indices, dim=1).values
                # print(y_preds.cpu().numpy())
                # print(y_true.cpu().numpy())
                loss1 = bce(output, target)
                loss2 = iou(y_preds, y_true)
                loss = loss1 - 10 * loss2
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_count += data.size(0)

            # log losses and metrics
            if isinstance(self.criterion, tuple):
                try:
                    batch_metrics = {"train/loss": loss.item(), "train/loss_bce": loss1.item(), "train/loss_iou": loss2.item()}
                except:
                    batch_metrics = {"train/loss": 0, "train/loss_bce": 0, "train/loss_iou": 0}
                    print(target.cpu().numpy())
            else:
                batch_metrics = {"train/loss": loss.item()}
            for name, metric in self.metrics.items():
                output, target = output.cpu(), target.cpu()
                if name == "PrecisionPerClass":
                    mat = metric(output, target)
                    for i, (tp, fp) in enumerate(mat):
                        batch_metrics[f"train/precision_{i}"] = tp / (tp + fp) if tp + fp > 0 else 0
                elif name in ["IoU", "Accuracy"]:
                    batch_metrics[f"train/{name}"] = metric(output, target)
                else:
                    raise NotImplementedError
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
            val_metric = self._valid_epoch(epoch)

        # learning rate scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        wandb.log({"train/lr": self.lr_scheduler.get_last_lr()[0]}, step=self.step)
        
        return train_loss, val_metric

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        
        running_loss = 0.0
        running_count = 0
        
        running_metrics = {}
        for name in self.metrics:
            if name == "PrecisionPerClass":
                for i in range(self.config.config["num_classes"]):
                    running_metrics[f'val/precision_{i}_fp'] = 0
                    running_metrics[f'val/precision_{i}_tp'] = 0
            else:
                running_metrics[f'val/{name}'] = 0
            

        with torch.no_grad():
            for data, target in tqdm(
                self.valid_data_loader,
                desc=f"VAL {epoch}/{self.epochs}",
                ncols=100,
            ):
                # move data to device
                data, target = data.to(self.device, dtype=torch.float32), target.to(self.device)

                # predictions and loss
                output = self.model(data)
                if isinstance(self.criterion, tuple):
                    bce, iou = self.criterion
                    y_preds = torch.sort(torch.topk(output.clone(), 4).indices, dim=1).values
                    y_true = torch.sort(torch.topk(target.clone(), 4).indices, dim=1).values
                    loss1 = bce(output, target)
                    loss2 = iou(y_preds, y_true)
                    loss = loss1 + 10 * loss2
                else:
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                running_count += data.size(0)
                
                for name, metric in self.metrics.items():
                    output, target = output.cpu(), target.cpu()
                    if name == "PrecisionPerClass":
                        mat = metric(output, target)
                        for i, (tp, fp) in enumerate(mat):
                            running_metrics[f'val/precision_{i}_tp'] += tp
                            running_metrics[f'val/precision_{i}_fp'] += fp
                    elif name in ["IoU", "Accuracy"]:
                        running_metrics[f'val/{name}'] += metric(output, target)
                    else:
                        raise NotImplementedError
            epoch_loss = running_loss / running_count
            
            # log losses and metrics
            running_metrics["val/loss"] = epoch_loss
            for name in self.metrics:
                if name == "PrecisionPerClass":
                    for i in range(self.config.config["num_classes"]):
                        tp, fp = running_metrics[f'val/precision_{i}_tp'], running_metrics[f'val/precision_{i}_fp']
                        running_metrics[f'val/precision_{i}'] = tp / (tp + fp) if tp + fp > 0 else 0
                        running_metrics.pop(f'val/precision_{i}_tp')
                        running_metrics.pop(f'val/precision_{i}_fp')
                else:
                    running_metrics[f'val/{name}'] /= len(self.valid_data_loader)
            wandb.log(running_metrics, step=self.step)

            if not self.config.with_wandb:
                print(f"VAL loss: {epoch_loss}")

        if self.mnt_metric == 'val_loss':
            return epoch_loss
        elif self.mnt_metric == 'val_iou':
            return running_metrics['val/IoU']
