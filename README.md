# Robustness Evaluation of RL-based Algorithms for Resource Allocation in 5G Network Slicing

In this project, I conduct a comprehensive evaluation of the robustness of Reinforcement Learning (RL)-based algorithms, specifically Deep Q-Network (DQN) and Dueling DQN, for dynamic resource allocation within the context of 5G network slicing. My objective is to develop a framework that not only meets diverse service requirements but also adapts to fluctuating network conditions, ensuring optimal utilization of network resources.

## Project Overview

5G network slicing enables the creation of multiple virtual networks, each tailored to meet the demands of different applications and services over a common physical infrastructure. The dynamic and heterogeneous nature of 5G services necessitates intelligent and flexible resource allocation strategies. This project explores the application of DQN and Dueling DQN algorithms in addressing these challenges, leveraging the strengths of RL for efficient decision-making in complex, variable environments.

## Methodology

### Reinforcement Learning Algorithms

- **Deep Q-Network (DQN):** DQN combines Q-learning with deep neural networks, providing the capability to handle high-dimensional state spaces, making it well-suited for the complex decision-making required in network slicing.
  
- **Dueling DQN:** An extension of DQN that separately estimates state values and the advantages of each action, leading to more precise policy evaluation and faster convergence, particularly beneficial for resource allocation's nuanced decision-making processes.

### Simulation Environment

To simulate the dynamic environment of 5G network slicing, we created a custom environment using OpenAI Gym, a toolkit for developing and comparing reinforcement learning algorithms. This environment replicates the real-world scenarios of network slicing, including varying service demands, resource constraints, and the need for real-time adaptation.


## Acknowledgement

This project's environment is inspired by the work presented in the following paper:

> Yuxiu Hua, Rongpeng Li, Zhifeng Zhao, Xianfu Chen, Honggang Zhang, "GAN-powered Deep Distributional Reinforcement Learning for Resource Management in Network Slicing," in *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 2, pp. 334–349, 2019, IEEE.


