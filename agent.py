#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/9 13:28
# @Author : ZhangKuo

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from deep_q_learning import DQN
from replay_memory import ReplayMemory


class Agent:
    def __init__(
        self,
        env,
        input_dim=3,
        output_dim=4,
        model_name="",
        memory_capacity=10000,
        batch_size=32,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=200,
        target_update=10,
        num_episodes=1000,
    ):
        self.env = env
        self.model_name = model_name
        self.memory = ReplayMemory(memory_capacity)
        self.writer = SummaryWriter()
        # 超参数设置
        self.batch_size = batch_size
        self.gamma = gamma
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps_done = 0
        self.num_episodes = num_episodes
        # 部署网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim).to(
            self.device
        )
        self.policy_net.load_state_dict(
            torch.load(self.model_name)
        ) if self.model_name else None
        self.target_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.loss = nn.SmoothL1Loss()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [random.randrange(self.output_dim)],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.cat(dones).to(self.device)
        # 获取当前状态动作值
        state_action_values = self.policy_net(states).gather(1, actions)
        # 计算target_net的最大动作值
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = rewards + self.gamma * next_state_values * (
            1 - dones
        )
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪用来防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        for episode in tqdm(range(self.num_episodes)):
            state = self.env.reset()
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            for t in range(10000):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(
                    [next_state], device=self.device, dtype=torch.float32
                )
                done = torch.tensor([done], device=self.device, dtype=torch.uint8)
                # 记录训练数据
                self.writer.add_scalar("reward", reward, global_step=episode)
                # 存储训练数据
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                self.optimize_model()
                if done:
                    break
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Complete")
        self.env.render()
        self.env.close()
        self.writer.close()
        torch.save(self.policy_net.state_dict(), f="./model/{self.num_episodes}.pth")
