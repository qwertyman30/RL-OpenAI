from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

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

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).    
    X_train = rgb2gray(X_train)
    X_train = np.concatenate([np.zeros((1, 96, 96)), X_train])
    X_train = np.array([X_train[i:i+history_length+1].T for i in range(len(X_train) - history_length)])
    
    X_valid = rgb2gray(X_valid)
    X_valid = np.concatenate([np.zeros((1, 96, 96)), X_valid])
    X_valid = np.array([X_valid[i:i+history_length+1].T for i in range(len(X_valid) - history_length)])
    
    y_train = np.array([action_to_id(y) for y in y_train])
    y_valid = np.array([action_to_id(y) for y in y_valid])
    return X_train, y_train, X_valid, y_valid

def train_model(X_train, y_train, X_valid, epochs, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    agent = BCAgent(lr=lr, history_length=1)
    tensorboard_eval = Evaluation(tensorboard_dir)
    X_train, y_train = shuffle(X_train, y_train)

    t_losses, t_accs, v_accs = [], [], []
    batches = int(len(X_train) / batch_size)
    val_batch_size = int(len(y_valid) / batches)
    
    for i in range(epochs):
        train_corr, val_corr = 0, 0
        for j in range(batches):
            X_batch_tr = X_train[j * batch_size:(j+1) * batch_size]
            y_batch_tr = y_train[j * batch_size:(j+1) * batch_size]

            X_batch_va = X_valid[j * val_batch_size:(j+1) * val_batch_size]
            y_batch_va = y_valid[j * val_batch_size:(j+1) * val_batch_size]
            X_batch_va = torch.Tensor(X_batch_va).cuda()
            X_batch_va = X_batch_va.view((-1, 1+history_length, 96, 96))

            t_loss, train_preds = agent.update(X_batch_tr, y_batch_tr)
            t_losses.append(t_loss)
            train_preds = torch.max(train_preds.data, 1)[1]
            train_corr += sum(train_preds.cpu().numpy() == y_batch_tr)
            with torch.no_grad():
                val_preds = agent.predict(X_batch_va)
                val_preds = torch.max(val_preds.data, 1)[1]
                val_corr += sum(val_preds.cpu().numpy() == y_batch_va)
            torch.cuda.empty_cache()

        train_acc = 100. * train_corr / len(X_train)
        val_acc = 100. * val_corr / len(X_valid)
        t_accs.append(train_acc)
        v_accs.append(val_acc)
        if i % 10 == 0:
            print(f"Epoch: {i+1}\tTrain Loss: {t_loss:.3f}\tTrain Accuracy:{train_acc:.3f}\tValidation accuracy:{val_acc:.3f}")
            tensorboard_eval.write_episode(i, { "Train Accuracy": train_acc, "Validation Accuracy": valid_acc })

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)
    return t_losses, t_accs, v_accs      


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, epochs=100, batch_size=288, lr=0.001)