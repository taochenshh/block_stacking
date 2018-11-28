import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import tensorflow as tf
import sys
import os
import glob
import copy

def main():
    FONT_SIZE = 22
    rcParams.update({'figure.autolayout': True,
                     'legend.fontsize': 17})
    data_dir = './data'
    log_dir = os.path.join(data_dir, 'logs/tb')
    event_file = glob.glob(os.path.join(log_dir, '*/events*'))[0]
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for e in tf.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'val/acc':
                val_acc.append(v.simple_value)
            elif v.tag == 'val/loss':
                val_loss.append(v.simple_value)
            elif v.tag == 'train/acc':
                train_acc.append(v.simple_value)
            elif v.tag == 'train/loss':
                train_loss.append(v.simple_value)
    x = np.arange(len(train_acc)) + 1
    acc_fig, acc_ax = plt.subplots()
    loss_fig, loss_ax = plt.subplots()
    acc_ax.plot(x, val_acc, 'b', label='Validation')
    acc_ax.plot(x, train_acc, 'r', label='Train')
    loss_ax.plot(x, val_loss, 'b', label='Validation')
    loss_ax.plot(x, train_loss, 'r', label='Train')

    acc_ax.ticklabel_format(axis='x', fontsize=FONT_SIZE)
    acc_ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    acc_ax.set_xlabel('Epoch', fontsize=FONT_SIZE)
    acc_ax.set_ylabel('Accuracy', fontsize=FONT_SIZE)
    acc_ax.set_title('Accuracy', fontsize=FONT_SIZE)
    acc_ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    acc_ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    leg = acc_ax.legend(loc=4)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    loss_ax.ticklabel_format(axis='x', fontsize=FONT_SIZE)
    loss_ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    loss_ax.set_xlabel('Epoch', fontsize=FONT_SIZE)
    loss_ax.set_ylabel('Loss', fontsize=FONT_SIZE)
    loss_ax.set_title('Loss', fontsize=FONT_SIZE)
    loss_ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    loss_ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    leg = loss_ax.legend(loc='upper right')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    acc_fig.savefig('accuracy.pdf', format='pdf')
    loss_fig.savefig('loss.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    main()