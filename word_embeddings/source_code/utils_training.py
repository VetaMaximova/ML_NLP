import torch
import numpy as np
import matplotlib.pyplot as plt

def binarization_regularization(model):  # binarization_regularization, weight_binarizer, 
    reg = torch.tensor(0.0)
    param_cnt = torch.tensor(0)
    for name, p in model.named_parameters():
        if 'weight' in name: 
            param_cnt += 1
            reg += torch.mean( torch.min( torch.abs(p), torch.abs(torch.abs(p)-1.0) ) )
    return reg/param_cnt

class NoamOpt:
    r'''Optim wrapper that implements adaptive learning rate.
    The source code is taken form: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    TODO: 
    1.  add grad clip
    2.  early stopping.
    '''

    def __init__(self, hidden_size, parameters, factor=0.2, warmup=5, args=None):
        self.optimizer = torch.optim.Adam(parameters, lr=0, betas=(0.9,0.98),eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = hidden_size
        self._rate = 0
        if args == None:
            self.verbose = 100
        else:
            args.verbose
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        self._rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = self._rate
        self.optimizer.step()

    def step_and_zero_grad(self):
        self.step()
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        'Implement learning rate above'
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def help(self,verbose=0):  
        sentence = ''
        if verbose < 100:
            sentence += 'Implements adaptive learning rate.'
        return sentence

    def plot_learning_rate(self):
        step = 20*self.warmup
        fig=plt.figure(figsize=(10,5))
        plot = plt.plot(np.arange(1, step), [self.rate(i) for i in range(1, step)])
        plt.xlabel('Step')
        plt.ylabel('Learning rate')

        return fig

    def __repr__(self):
        return self.help(self.verbose)           
    
    def state_dict(self):
        state = {}
        state['_step'] = self._step
        state['optimizer'] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, optim_state):
        self._step = optim_state['_step']
        self.optimizer.load_state_dict(optim_state['optimizer'])

class ScheduledOptim():
    r'''A simple wrapper class for learning rate scheduling.
    Taken from the pytorch version of BERT, from:
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/optim_schedule.py
    Modified by AKF. Okey, this is the same as NoamOpt. This should be removed.
    '''
    # optimizer = utils_training.ScheduledOptim(optimizer=torch.optim.Adam(baseline_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),\
    #                                     d_model=hidden_layer_dim, n_warmup_steps=100, args=args )

    def __init__(self, optimizer, d_model, n_warmup_steps, args):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.verbose = args.verbose

    def step_and_zero_grad(self):
        self.step_and_update_lr()
        self._optimizer.zero_grad()

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def help(self,verbose=0):  
        sentence = ''
        if verbose < 100:
            sentence += 'Implements adaptive learning rate as it was used to train BERT.'
        return sentence

    def plot_learning_rate(self):
        # print(self.init_lr)
        # self.n_current_steps += 1
        # print(self._get_lr_scale())
        step = 5*self.n_warmup_steps
        fig=plt.figure(figsize=(10,5))
        # print(step)
        lr = [self._update_learning_rate() for i in range(1, step)]
        # print(lr)
        plt.plot(np.arange(1, step), lr)
        self.n_current_steps = 0
        plt.xlabel('Step')
        plt.ylabel('Learning rate')

        return fig

    def __repr__(self):
        return self.help(self.verbose)                    
