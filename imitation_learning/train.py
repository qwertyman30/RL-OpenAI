from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import time
import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    return X, y

def preprocessing(X, y, history_length=1):
    print('Preprocessing data')

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).    
    X = rgb2gray(X)
    X = np.concatenate([np.zeros((history_length, 96, 96)), X])
    X = np.array([X[i:i+history_length+1].T for i in range(len(X) - history_length)])
    
    y = np.array([action_to_id(label) for label in y])
    return X, y

def train_val_split(X, y, frac=0.1):
    n_samples = len(X)
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

def train_model(X_train, y_train, X_valid, y_valid, epochs, batch_size, lr, history_length=1, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    agent = BCAgent(lr=lr, history_length=history_length)
    tensorboard_eval = Evaluation(tensorboard_dir)

    t_losses, t_accs, v_accs = [], [], []
    train_batch_size = int(len(X_train) / batches)
    val_batch_size = int(len(y_valid) / batches)
    
    best_val_acc = 0
    
    for i in range(epochs):
        start = time.time()
        train_corr, val_corr = 0, 0
        for j in range(batches):
            X_batch_tr = X_train[j * train_batch_size:(j+1) * train_batch_size]
            y_batch_tr = y_train[j * train_batch_size:(j+1) * train_batch_size]

            X_batch_va = X_valid[j * val_batch_size:(j+1) * val_batch_size]
            y_batch_va = y_valid[j * val_batch_size:(j+1) * val_batch_size]
            X_batch_va = torch.Tensor(X_batch_va).cuda()
            X_batch_va = X_batch_va.view((-1, 1+history_length, 96, 96))

            t_loss, train_preds = agent.update(X_batch_tr, y_batch_tr)
            train_preds = torch.max(train_preds.data, 1)[1]
            train_corr += sum(train_preds.cpu().numpy() == y_batch_tr)
            with torch.no_grad():
                val_preds = agent.predict(X_batch_va)
                val_preds = torch.max(val_preds.data, 1)[1]
                val_corr += sum(val_preds.cpu().numpy() == y_batch_va)
            torch.cuda.empty_cache()

        train_acc = 100. * train_corr / len(X_train)
        val_acc = 100. * val_corr / len(X_valid)
        if best_val_acc < val_acc:
            print(f'Saving model at epoch {i+1}')
            agent.save(f'agent1_15k_epoch{i+1}_{batches}.pt')
            best_val_acc = val_acc
        
        t_losses.append(t_loss)
        t_accs.append(train_acc)
        v_accs.append(val_acc)
        
        if i % 10 == 0:
            print(f"Epoch: {i+1}\tTrain Loss: {t_loss:.3f}\tTrain Accuracy:{train_acc:.3f}\tValidation accuracy:{val_acc:.3f}")
            tensorboard_eval.write_episode(i, { "Train Accuracy": train_acc, "Validation Accuracy": valid_acc })
        end = time.time()
        print(f'Epoch {i+1} Time: {end - start}s')
    return t_losses, t_accs, v_accs


if __name__ == "__main__":

    # read data
    X, y = read_data("./data")

    # preprocess data
    X, y = preprocessing(X, y, history_length=3)

    print("Upsample data")
    X_0 = X[y == 0]
    lenx0 = len(X_0)

    X = np.concatenate([
        X_0,
        resample(X[y == 1], replace=True, n_samples=lenx0),
        resample(X[y == 2], replace=True, n_samples=lenx0),
        resample(X[y == 3], replace=True, n_samples=lenx0)
    ])

    y = np.concatenate([
        np.zeros((lenx0)),
        np.ones(lenx0),
        np.ones(lenx0)*2,
        np.ones(lenx0)*3
    ])

    print("Shuffle data")
    X, y = shuffle(X, y)

    print("Train val split")
    X_train, y_train, X_valid, y_valid = train_val_split(X, y)

    del X
    del y

    # train model (you can change the parameters!)
    losses, t_accs, v_accs = train_model(X_train, y_train, X_valid, y_valid, history_length=3, epochs=40, batches=200, lr=0.001)
    
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss vs epochs')
    plt.savefig('Training loss vs epochs.png')
    plt.show()

    plt.plot(t_accs, label='Training')
    plt.plot(v_accs, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs epochs')
    plt.legend()
    plt.savefig('Accuracy vs epochs.png')
    plt.show()
