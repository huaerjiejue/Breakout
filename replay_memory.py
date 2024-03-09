#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/9 11:04
# @Author : ZhangKuo
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
