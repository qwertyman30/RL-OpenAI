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

    t_losses, t_f1_scores, v_f1_scores = [], [], []
    train_batch_size = int(len(X_train) / batches)
    val_batch_size = int(len(y_valid) / batches)
    
    best_f1_score = 0
    
    for i in range(epochs):
        start = time.time()
        all_train_preds, all_val_preds = [], []
        for j in range(batches):
            X_batch_tr = X_train[j * train_batch_size:(j+1) * train_batch_size]
            y_batch_tr = y_train[j * train_batch_size:(j+1) * train_batch_size]

            X_batch_va = X_valid[j * val_batch_size:(j+1) * val_batch_size]
            y_batch_va = y_valid[j * val_batch_size:(j+1) * val_batch_size]
            X_batch_va = torch.Tensor(X_batch_va).cuda()
            X_batch_va = X_batch_va.view((-1, 1+history_length, 96, 96))

            t_loss, train_preds = agent.update(X_batch_tr, y_batch_tr)
            train_preds = torch.max(train_preds.data, 1)[1]
            all_train_preds = np.concatenate([all_train_preds, train_preds.cpu().numpy()])
            with torch.no_grad():
                val_preds = agent.predict(X_batch_va)
                val_preds = torch.max(val_preds.data, 1)[1]
                all_val_preds = np.concatenate([all_val_preds, val_preds.cpu().numpy()])
            torch.cuda.empty_cache()

        if (j+1)*train_batch_size < len(X_train):
            X_batch_tr = X_train[(j+1) * train_batch_size:]
            y_batch_tr = y_train[(j+1) * train_batch_size:]

            X_batch_va = X_valid[(j+1) * val_batch_size:]
            y_batch_va = y_valid[(j+1) * val_batch_size:]
            X_batch_va = torch.Tensor(X_batch_va).cuda()
            X_batch_va = X_batch_va.view((-1, 1+history_length, 96, 96))

            t_loss, train_preds = agent.update(X_batch_tr, y_batch_tr)
            train_preds = torch.max(train_preds.data, 1)[1]
            all_train_preds = np.concatenate([all_train_preds, train_preds.cpu().numpy()])
            with torch.no_grad():
                val_preds = agent.predict(X_batch_va)
                val_preds = torch.max(val_preds.data, 1)[1]
                all_val_preds = np.concatenate([all_val_preds, val_preds.cpu().numpy()])

        train_f1_score = f1_score(all_train_preds, y_train, average='weighted')
        val_f1_score = f1_score(all_val_preds, y_valid, average='weighted')
        if best_f1_score < val_f1_score:
            print(f'Saving model at epoch {i+1}')
            agent.save(f'agent1_15k_epoch{i+1}_{batches}.pt')
            best_f1_score = val_f1_score

        t_losses.append(t_loss)
        t_f1_scores.append(train_f1_score)
        v_f1_scores.append(val_f1_score)

        if i % 10 == 0:
            print(f"Epoch: {i+1}\tTrain Loss: {t_loss:.3f}\tTrain f1_score:{train_f1_score:.3f}\tValidation f1_score:{val_f1_score:.3f}")
            tensorboard_eval.write_episode(i, { "Train Accuracy": train_acc, "Validation Accuracy": valid_acc })
        end = time.time()
        print(f'Epoch {i+1} Time: {end - start}s')
    return t_losses, t_f1_scores, v_f1_scores


if __name__ == "__main__":
    # read data
    X, y = read_data(".\data")

    # preprocess data
    X, y = preprocessing(X, y, history_length=3)
    
    print("Upsampling data")
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    X_2 = X[y == 2]
    X_3 = X[y == 3]
    X_4 = X[y == 4]

    X = np.concatenate([
        X_0,
        resample(X_1, replace=True, n_samples=len(X_1)*2),
        resample(X_2, replace=True, n_samples=len(X_2)*3),
        resample(X_3, replace=True, n_samples=len(X_3)*2),
        resample(X_4, replace=True, n_samples=len(X_4)*6)
    ])

    y = np.concatenate([
        np.zeros((len(X_0))),
        np.ones((len(X_1)*2)),
        np.ones(len(X_2)*3)*2,
        np.ones(len(X_3)*2)*3,
        np.ones(len(X_4)*6)*4
    ])

    print("Shuffle data")
    X, y = shuffle(X, y)

    print("Train val split")
    X_train, y_train, X_valid, y_valid = train_val_split(X, y)

    # Lack of ram. so delete unnecessary variables
    del X
    del y
    del X_0
    del X_1
    del X_2
    del X_3
    del X_4

    # train model (you can change the parameters!)
    losses, t_f1_scores, v_f1_scores = train_model(X_train, y_train, X_valid, y_valid, history_length=3, epochs=100, batches=50, lr=0.001)

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss vs epochs')
    plt.savefig('Training loss vs epochs.png')
    plt.show()

    plt.plot(t_f1_scores, label='Training')
    plt.plot(v_f1_scores, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs epochs')
    plt.legend()
    plt.savefig('F1 Score vs epochs.png')
    plt.show()