import os, copy
import json
import nltk
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from functools import reduce
import sys
import psutil
import gc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D    
import matplotlib
import random
import matplotlib.style as style
import subprocess

def get_gpu_memory_usage():   
    used_mem = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    free_mem = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    
    print("CPU has {0}Mb free, {1}Mb used".format(float(free_mem), float(used_mem)))


def get_device(force_cpu = False):
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        print('Found device:', device)
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if (not torch.cuda.is_available() or force_cpu):
        device = torch.device('cpu')
        print('Found device:', device)
        print('CPU cores count: {0}'.format(psutil.cpu_count()))
        torch.set_default_tensor_type('torch.FloatTensor')

    print('Using device:', device)        
    print('sys version: {0}'.format(sys.version)) 
    return device

  
def cpuStats(): 
    pid = os.getpid() 
    py = psutil.Process(pid) 
    process = psutil.Process(py.pid)
    mem_info = process.memory_info()
    data = {"pid": process.pid,
            "run on cpu": process.cpu_num(),
            "percent_cpu_used": process.cpu_percent(interval=1.0),
            "percent_physical system memory_used": process.memory_percent()
           }

    print(data)
    
def plot_grad_flow_(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
   
    
def plot_grad_flow_add_epoch(named_parameters, ave_grads, std_grads, max_y):
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if (max_y < p.grad.abs().mean()):
                max_y = p.grad.abs().mean()
            if (max_y < p.grad.abs().std()):
                max_y = p.grad.abs().std()
            ave_grads[p].append(p.grad.abs().mean())
            std_grads[p].append(p.grad.abs().std())
            
    return max_y
                      
            
def plot_grad_flow(named_parameters, epochs, ave_grads, std_grads, max_y):
    cnt_grads_on_plot = 5
    num = 0
    label_added = False
    style.use('seaborn')
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            plt.plot(epochs, ave_grads[p], label='avg for {0}'.format(n))
            plt.plot(epochs, std_grads[p], label='std for {0}'.format(n), linestyle='--')
            num = num + 1
            label_added = True
            if (num % cnt_grads_on_plot == 0):
                plt.ylim(ymin=0,ymax=(max_y))
                plt.xlim(xmin=0, xmax=len(epochs) - 1)
                plt.hlines(0, 0, len(epochs), linewidth=1, color="k" )
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel("epochs")
                plt.ylabel("average/std gradient")
                plt.title("Gradient flow")
                plt.grid(True)
                plt.figure(num / cnt_grads_on_plot + 1)
    if not label_added:
        plt.ylim(ymin=0,ymax=(max_y))
        plt.xlim(xmin=0, xmax=len(epochs) - 1)
        plt.hlines(0, 0, len(epochs), linewidth=1, color="k" )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("epochs")
        plt.ylabel("average/std gradient")
        plt.title("Gradient flow")
        plt.grid(True)