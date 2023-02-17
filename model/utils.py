import matplotlib.pyplot as plt
import numpy as np
# font = {
#         'weight' : 'light',
#         'size'   : 12}

# plt.rc('font', **font)

def plot_acc(acc_train,acc_val):
    fig, ax = plt.subplots()
    step = len(acc_train) // 10
    if len(acc_train) <= 10:
        step = 1
    ax.plot(acc_train, label = 'train',color = 'b')
    ax.plot(acc_val, label = 'val',color = 'g')
    ax.set_ylabel('Acc',fontsize=15)
    ax.set_xlabel('Epoch',fontsize=15)
    ax.set_title('Accuracy training')
    ax.set_xticks(np.arange(0,len(acc_train)+1,step))
    ax.set_xticklabels(np.arange(1,len(acc_train)+1+1,step))
    ax.grid()
    ax.legend()
    plt.tight_layout()
    ax.figure.savefig('/home/tinhnv/Cty/classifier_flip/output15022023/plot_accuracy_training.jpg')
    plt.close('all') 

def plot_loss(loss_train,loss_val):
    fig, ax = plt.subplots()
    step = len(loss_train) // 10
    if len(loss_train) <= 10:
        step = 1
    ax.plot(loss_train, label = 'train',color = 'b')
    ax.plot(loss_val, label = 'val',color = 'g')
    ax.set_ylabel('Loss',fontsize=15)
    ax.set_xlabel('Epoch',fontsize=15)
    ax.set_title('Loss training')
    ax.set_xticks(np.arange(0,len(loss_train)+1,step))
    ax.set_xticklabels(np.arange(1,len(loss_val)+2,step))
    ax.grid()
    ax.legend()
    plt.tight_layout()
    ax.figure.savefig('/home/tinhnv/Cty/classifier_flip/output15022023/plot_loss_training.jpg')
    plt.close('all') 