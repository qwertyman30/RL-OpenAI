#import tensorflow as tf
import numpy as np
from collections import namedtuple
import torch
from replay_buffer import ReplayBuffer

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss().cuda()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        replay_size = len(self.replay_buffer._data.states)
        
        if replay_size < self.batch_size:
            return

        states, actions, next_states, rewards, terminals = self.replay_buffer.next_batch(self.batch_size)
        td_targets = []

        for i in range(self.batch_size):
            td_target = self.predict_Q(states[i])
            if terminals[i]:
                td_target[actions[i]] = rewards[i]
            else:
                td_target_next = self.predict_Q_target(next_states[i])
                td_target[actions[i]] = reward + self.gamma * torch.max(td_target_next).item()

            td_targets.append(td_target)

        td_targets = [t.detach().cpu().numpy() for t in td_targets]
        td_targets = np.vstack(td_targets)
        td_targets = torch.Tensor(td_targets).cuda()

        y_pred = self.predict_Q(states)
        loss = self.loss_function(y_pred, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            state = torch.Tensor(state).cuda()
            q_values = self.Q(state)
            action_id = torch.argmax(q_values).item()
        else:
            action_id = np.random.choice(np.arange(0, self.num_actions), p=[0.5, 0.5])
        return action_id

    def predict_Q_target(self, state):
        return self.Q_target(torch.Tensor(state).cuda())
    
    def predict_Q(self, state):
        return self.Q(torch.Tensor(state).cuda())

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
