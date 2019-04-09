from typing import Optional, Dict
import torch
from oai import logger

log = logger.global_logger(__name__)
from oai.tabular.datasets.boilerplate import DualHeadDataset, collate_fn

import pandas as pd

from pathlib import Path
import json, _jsonnet

class Learner(object):
    """docstring for Learner"""
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim, 
                 criterion,
                 metrics,
                 batch_size,
                 dataloaders: dict, 
                 state: Optional[Dict] = None,
                 history: Optional[list] = None,
                 device='cpu',
                 num_workers=1,
                 pin_memory=True,
                 pretty_print=True,
                 multi_gpu=False):
        super(Learner, self).__init__()
        if multi_gpu is True:
            print('GPU count:', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        if device == 'cuda':
            self.model = model.cuda()  # to(device)
            print(f'Model is on GPU: {next(model.parameters()).is_cuda}')
        else:
            print('GPU not utilized')
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders

        # Defined in arch
        self.batch_size = batch_size
        self.metrics = metrics # ...
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if state is None:
            self.state = {
                'epoch': 0,
                'loss': 0.0,
                'val_loss': 0.0,
                'metric': 0.0,
                'val_metric': 0.0,
                'finite_state_machine': 'learning' # stopped, ....
            }
        else:
            self.state = state

        if history is None:
            self.history = []
            self.history.append(self.state)
        else:
            self.history = history

        self.pretty_print = pretty_print
        self.optimizer = self.optimizer(self.model.parameters(), lr=0.00001)   # put in __init__
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        
    #@classmethod
    def hydrate(checkpoint_path: str):
        """Load model, optimizer and state"""
        checkpoint = torch.load(path)
        self.model.update(checkpoint['model'])
        self.optimizer.update(checkpoint['optimizer'])
        self.scheduler.update(checkpoint['scheduler'])

        self.state = checkpoint['state']
        self.history = checkpoint['history']
    
    def checkpoint(self, save_dir: str):
        """save __module__, __name__, .state_dicts, epoch, state, ...."""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        base_file_name = f'{self.state["epoch"]}_{self.val_loss_epoch:.4f}'
        checkpoint_file_path = save_dir / f'{base_file_name}_checkpoint.pth'
        self.checkpoint_file_path = checkpoint_file_path
        history_file_path = save_dir / f'{base_file_name}_history.json'

        checkpoint = {
            'state': self.state,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'history': self.history
        }

        torch.save(checkpoint, checkpoint_file_path)
        #torch.jit.save(self.model, str(model_file_path))

        with open(history_file_path, 'w') as f:
            json.dump(self.history, f)
    
    def _update_state(self):

        increment_epoch = self.state['epoch'] + 1
        self.state = {
            'epoch': increment_epoch,
            'loss': self.loss_epoch,
            'val_loss': self.val_loss_epoch,
            'metrics': self.metrics_epoch,
            'val_metrics': self.val_metrics_epoch
        }
        self.history.append(self.state)

        if self._continue_learning() is True:
            self.state['finite_state_machine'] = 'learning'
        else:
            self.state['finite_state_machine'] = 'stopped'

    def _continue_learning(self) -> bool:
        epochs = [x['epoch'] for x in self.history]
        losses = [x['val_loss'] for x in self.history]
        data = {'epoch': epochs, 'loss': losses}
        df = pd.DataFrame(data).sort_values('epoch', ascending=False)

        current_loss = df.head(1).loss.values
        past_20_loss = df.head(21).tail(20).loss.values

        if df.shape[0] < 20 or (current_loss < past_20_loss).any():
            return True
        else:
            return False

    def learn(self) -> float:
        self.loss_accumulate = 0.0
        self.metrics_accumulate = {k: 0.0 for k in self.metrics.keys()}
        
        self.val_loss_accumulate = 0.0
        self.val_metrics_accumulate = {k: 0.0 for k in self.metrics.keys()}


        self.model.train()
        for batch_idx, batch in enumerate(self.dataloaders['learn'].generate_batches(batch_size=self.batch_size, 
                                                                                     collate_fn=collate_fn,
                                                                                     device=self.device,
                                                                                     num_workers=self.num_workers,
                                                                                     pin_memory=self.pin_memory)):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            yhat = self.model(cat_data=batch['cat_X'], cont_data=batch['cont_X'])
            ys = [yhat, batch['y']]
            
            loss = self.criterion(*ys)
            self.loss_accumulate += loss.item()
                        
            for k, v in self.metrics.items():
                self.metrics_accumulate[k] += self.metrics[k](*ys).item()
            
            loss.backward()
            self.optimizer.step()
        
        self.metrics_epoch = {k: v / (batch_idx + 1) for k, v in self.metrics_accumulate.items()}
        self.loss_epoch = self.loss_accumulate / (batch_idx + 1)
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val'].generate_batches(batch_size=self.batch_size, 
                                                                                       collate_fn=collate_fn,
                                                                                       device=self.device,
                                                                                       num_workers=self.num_workers)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                val_yhat = self.model(cat_data=batch['cat_X'], cont_data=batch['cont_X']) 

                val_loss = self.criterion(val_yhat, batch['y'])

                self.val_loss_accumulate += val_loss.item()
                for k, v in self.metrics.items():
                    self.val_metrics_accumulate[k] += self.metrics[k](val_yhat, batch['y']).item()
        
        self.val_metrics_epoch = {k: v / (batch_idx + 1) for k, v in self.val_metrics_accumulate.items()}
        self.val_loss_epoch = self.val_loss_accumulate / (batch_idx + 1)

        self.scheduler.step(self.val_loss_epoch)
        self._update_state()

        if self.pretty_print is True:
            self._pretty_print()
        return self.val_loss_epoch

    def test(self) -> float:
        self.model.eval()

        for batch_idx, batch in enumerate(self.dataloaders['test']):

            val_yhat = self.model(cat_data=batch['cat_X'],
                                  cont_data=batch['cont_X'])

            val_loss = self.criterion(val_yhat, batch['y'])
            val_metric = self.compute_metrics(val_yhat, batch['y'])

        return val_loss
    
    def _pretty_print(self):
        printable_loss = f'loss: {self.loss_epoch:.2f}'
        printable_val_loss = f'val_loss: {self.val_loss_epoch:.2f}'

        printable_metrics = printable_metrics_fct(self.metrics_epoch)
        printable_val_metrics = printable_metrics_fct(self.val_metrics_epoch, val=True)

        print(f'Epoch: {self.state["epoch"]}, {printable_loss}, {printable_val_loss}, {printable_metrics}, {printable_val_metrics}')

def printable_metrics_fct(metric: dict, val=False) -> str:

    for k, v in metric.items():
        string = f'{k}: {v:.2f}'
        if val is True:
            string = f'val_{string}'
    return string