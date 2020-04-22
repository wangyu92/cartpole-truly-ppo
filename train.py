import os
import numpy as np
import torch
import torch.nn.functional as F
import gym
import datetime

from torch.utils.tensorboard import SummaryWriter
from model import Model

T_HORIZON       = 32
N_MINIBATCH     = 1
LEARNING_RATE   = 5e-5
EPS             = 0.2
N_EPOCH         = 20

GAMMA           = 0.99
LAMBDA          = 0.95

def compute_gae(rewards, values, values_n, dones):
    td_target = rewards + GAMMA * values_n * (1 - dones)
    delta = td_target - values
    gae = np.append(np.zeros_like(rewards), [0], axis=-1)
    for i in reversed(range(len(rewards))):
        gae[i] = GAMMA * LAMBDA * gae[i + 1] * (1 - dones[i]) + delta[i]
    gae = gae[:-1]
    return gae, td_target

def train_step(model, states, actions, probs, rewards, dones, statesn):
    device = model.getdevice()

    # compute return and advantages
    values = model.values(states)
    valuesn = model.values(statesn)
    advs, returns = compute_gae(rewards, values, valuesn, dones)
    
    # advs = returns - values

    # # Normalize the advantages
    # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    # convert from numpy to tensors
    states = torch.tensor(states.copy(), device=device).float()
    actions = torch.tensor(actions.copy(), device=device).long()
    probs = torch.tensor(probs.copy(), device=device).float()
    advs = torch.tensor(advs.copy(), device=device).float()
    returns = torch.tensor(returns.copy(), device=device).float()
    values = torch.tensor(values.copy(), device=device).float()

    # reshape the data
    actions = actions[:, None]
    probs = probs[:, None]
    advs = advs[:, None]
    returns = returns[:, None]
    values = values[:, None]

    # compute gradient and updates
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    ppreds, vpreds = model(states)
    loss_policy = loss_policy_fn(ppreds, actions, probs, advs)
    loss_value = loss_value_fn(vpreds, returns, values)
    vf_coef = 0.5

    loss = (loss_policy) + (vf_coef * loss_value)
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), loss_policy.detach().cpu().numpy(), loss_value.detach().cpu().numpy()

def loss_policy_fn(preds, actions, oldprobs, advs):
    """
    preds = (batch_size, action_size)
    actions = (batch_size, 1)
    oldprobs = (batch_size, 1)
    advs = (batch_size, 1)
    """
    probs = torch.gather(preds, dim=-1, index=actions)
    ratio = torch.exp(torch.log(probs) - torch.log(oldprobs))
    surr1 = ratio * advs
    surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advs

    loss = -torch.min(surr1, surr2)
    return loss.mean()

def loss_value_fn(preds, returns, oldvpred):
    vpredclipped = oldvpred + torch.clamp(preds - oldvpred, -EPS, EPS)
    vf_losses1 = (preds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2

    loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
    return loss.mean()

def test_model(model):
    model.eval()
    env = gym.make('CartPole-v1')

    score = [0.]

    for _ in range(20):
        s = env.reset()
        while True:
            a = model.action(s)
            s, r, d, _ = env.step(a)
            score[-1] += r
            if d:
                score.append(0.)
                break

    return np.mean(score)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # tensorboard
    s_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = '../tensorboard/' + 'CartPole-Truly-PPO_' + s_time
    train_summary_writer = SummaryWriter(dir_path)

    env = gym.make('CartPole-v1')
    model = Model(env.observation_space.shape[0], env.action_space.n)
    model.to(device)

    states = np.empty((T_HORIZON, env.observation_space.shape[0]))
    statesn = np.empty((T_HORIZON, env.observation_space.shape[0]))
    probs = np.empty((T_HORIZON,))
    actions = np.empty((T_HORIZON,), dtype=np.int32)
    rewards = np.empty((T_HORIZON,))
    dones = np.empty((T_HORIZON,))

    entropy = np.empty((T_HORIZON,))

    num_updates = 0
    epi_rewards = [0.]
    interval = 20

    sn = env.reset()
    while True:

        for t in range(T_HORIZON):
            states[t] = sn.copy()
            actions[t], probs[t], entropy[t] = model.action_sample(states[t])
            sn, rewards[t], dones[t], _ = env.step(actions[t])
            statesn[t] = sn.copy()

            epi_rewards[-1] += rewards[t]

            if dones[t]:
                sn = env.reset()
                statesn[t] = sn.copy()

                train_summary_writer.add_scalar('train_epi_rewards', epi_rewards[-1], num_updates)
                epi_rewards.append(0.)

        # train
        losses = []
        losses_actor = []
        losses_critic = []
        for _ in range(N_EPOCH): 
            loss, loss_p, loss_c = train_step(model, states, actions, probs, rewards, dones, statesn)
            losses.append(loss)
            losses_actor.append(loss_p)
            losses_critic.append(loss_c)
        train_summary_writer.add_scalar('Loss/Total', np.mean(losses), num_updates)
        train_summary_writer.add_scalar('Loss/Actor', np.mean(losses_actor), num_updates)
        train_summary_writer.add_scalar('Loss/Critic', np.mean(losses_critic), num_updates)
        train_summary_writer.add_scalar('Loss/Entropy', np.mean(entropy), num_updates)
        num_updates += 1

        score = test_model(model)
        train_summary_writer.add_scalar('test_epi_rewards', score, num_updates)

        if num_updates % interval == 0:
            print("num_epi = {}, num_updates = {} test score = {}".format(len(epi_rewards), num_updates, score))

        if np.mean(epi_rewards[-5:]) == 500.0:
            break

    train_summary_writer.close()