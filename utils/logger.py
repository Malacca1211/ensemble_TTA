# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig','writefile']

def writefile(fname, content):
    with open(fname, 'a') as f:
        f.write(content)


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None,alpha = 1):
    names = logger.names if names == None else names
    numbers = logger.numbers
    max_=[]
    min_=[]
    legends=[]
    for _, name in enumerate(names):
        try: 
            x = np.arange(len(numbers[name]))
        except:
            print(f'!!!!! {name} not in key for {logger.title} !!!!!')
            continue
        plt.plot(x, np.asarray([float(m) for m in numbers[name]]), alpha=alpha)
        max_.append(max(numbers[name]))
        min_.append(min(numbers[name]))
        legends.append(logger.title + '(' + name + ')')

    max_=max(max_)
    min_=min(min_)
        
    return legends,max_,min_

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        # plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None,alpha = 1,y_lim = None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        maxs=[]
        mins=[]
        for logger in self.loggers:
            legend_text_,max_,min_ = plot_overlap(logger, names,alpha = alpha)
            legend_text += legend_text_
            maxs.append(max_)
            mins.append(min_)
        # plt.yticks(np.arange(min(mins), max(maxs),20 ))
        # plt.legend(legend_text)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
                    
# if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    # paths = {
    # 'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
    # 'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    # 'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    # }

    # field = ['Valid Acc.']

    # monitor = LoggerMonitor(paths)
    # monitor.plot(names=field)
    # savefig('test.eps')