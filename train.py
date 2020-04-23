import os
import numpy as np
import torch
import torch.nn.functional as F
import gym
import datetime

from torch.utils.tensorboard import SummaryWriter
from model import Model

T_HORIZON       = 128
N_MINIBATCH     = 1
LEARNING_RATE   = 5e-4
EPS             = 0.2
N_EPOCH         = 5

GAMMA           = 0.99
LAMBDA          = 0.95

TYPE_PPO        = 0
TYPE_ROLLBACK   = 1
TYPE_TR         = 2
TYPE_TRULY      = 3
PPO_TYPE        = TYPE_PPO


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

    states = np.split(states, N_MINIBATCH)
    actions = np.split(actions, N_MINIBATCH)
    probs = np.split(probs, N_MINIBATCH)
    rewards = np.split(rewards, N_MINIBATCH)
    dones = np.split(dones, N_MINIBATCH)
    statesn = np.split(statesn, N_MINIBATCH)

    losses = []
    losses_p = []
    losses_v = []
    for i in range(N_MINIBATCH):
        values = model.values(states[i])
        valuesn = model.values(statesn[i])
        advs, returns = compute_gae(rewards[i], values, valuesn, dones[i])

        states[i] = torch.tensor(states[i].copy(), device=device).float()
        actions[i] = torch.tensor(actions[i].copy(), device=device).long()[:, None]
        probs[i] = torch.tensor(probs[i].copy(), device=device).float()[:, None]
        advs = torch.tensor(advs.copy(), device=device).float()[:, None]
        returns = torch.tensor(returns.copy(), device=device).float()[:, None]
        values = torch.tensor(values.copy(), device=device).float()[:, None]

        # compute gradient and updates
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        ppreds, vpreds = model(states[i])
        loss_policy = loss_policy_fn(ppreds, actions[i], probs[i], advs)
        loss_value = loss_value_fn(vpreds, returns, values)
        vf_coef = 0.5

        entropy = torch.distributions.Categorical(probs=ppreds)
        entropy = entropy.entropy().mean()

        loss = (loss_policy) + (vf_coef * loss_value) + (- 0.01 * entropy)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        losses_p.append(loss_policy.item())
        losses_v.append(loss_value.item())

    return np.mean(losses), np.mean(losses_p), np.mean(losses_v)

def loss_policy_fn(preds, actions, oldprobs, advs):
    probs = torch.gather(preds, dim=-1, index=actions)
    ratio = torch.exp(torch.log(probs) - torch.log(oldprobs))

    if PPO_TYPE == TYPE_PPO:
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advs
        loss = -torch.min(surr1, surr2)
    elif PPO_TYPE == TYPE_ROLLBACK:
        slope = -0.3
        pg_targets = torch.where(advs >= 0,
            torch.where(ratio <= 1 + EPS, ratio, slope * ratio + (1 - slope) * (1 + EPS)),
            torch.where(ratio >= 1 - EPS, ratio, slope * ratio + (1 - slope) * (1 - EPS))
        ) * advs
        loss = -pg_targets

    return loss.mean()

def loss_value_fn(preds, returns, oldvpred):
    vpredclipped = oldvpred + torch.clamp(preds - oldvpred, -EPS, EPS)
    vf_losses1 = F.smooth_l1_loss(preds, returns)
    vf_losses2 = F.smooth_l1_loss(vpredclipped, returns)

    # loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
    loss = vf_losses1
    return loss

def test_model(model):
    model.eval()
    env = gym.make('CartPole-v1')

    score = []

    for _ in range(10):
        s = env.reset()
        score.append(0.)
        while True:
            a = model.action(s)
            s, r, d, _ = env.step(a)
            score[-1] += r
            if d:
                break

    return sum(score)/len(score)

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
    epi_rewards = []
    interval = 20

    sn = env.reset()
    while True:
        epi_rewards.append(0.)
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

        # train
        model.train()

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

        if score == 500.0:
            break

    train_summary_writer.close()