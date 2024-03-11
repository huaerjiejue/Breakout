#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/3/11 12:53
# @Author : ZhangKuo
import gymnasium as gym

from agent import Agent

env = gym.make("Breakout-v5")
agent = Agent(env)
agent.train()
