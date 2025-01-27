import numpy as np
import torch

import math

# Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

# References:
# [1] https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        # joaofmari
        self.best_epoch = None

    # joaofmatri
    ### def __call__(self, val_loss, model):
    def __call__(self, val_loss, model, epoch):

        # if nan, we assume 1.0
        # if math.isnan(val_loss):
        #     val_loss = 1.0

        score = -val_loss

        if self.best_score is None:
            ### print('--> Primeira época...')
            self.best_score = score
            self.save_checkpoint(val_loss, model)

            # joaofmari
            self.best_epoch = epoch

        elif score < self.best_score + self.delta or math.isnan(val_loss):
            # Score diminuiu. Loss aumentou.
            ### print('--> Loss aumentou... Score diminuiu')
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            ### print('--> Loss diminuiu ou manteve... Score aumentou ou manteve... ')
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

            # joaofmari
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss