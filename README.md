# Accelerated Federated Learning Over MAC in Heterogeneous Networks


This repository is to analyze the convergence rate of a federated learning algorithm named SCAFFOLD which is inspired by the SVRG algorithm in noisy fading MAC settings and heterogenous data, in order to formulate a new algorithm that accelerates the learning process in such settings.  We inspired by 3 related works:

1.	**On Analog Gradient Descent Learning Over Multiple Access Fading Channels
The authors implemented GBMA algorithm in which the users transmit an analog function of their local gradient using a common shaping-waveform and the network edge update the global model using a received superposition of the analog transmitted signals which represents a noisy distorted version of the gradients.
https://arxiv.org/abs/1908.07463

2.	**Over-the-Air Federated Learning from Heterogeneous Data
The authors introduce time-varying pre-coding and scaling scheme COTAF which facilitates the aggregation and gradually mitigates the noise effect and maintains the convergence properties of local SGD with heterogeneous data across users.
https://arxiv.org/abs/2009.12787

3.	**SCAFFOLD - Stochastic Controlled Averaging for Federated Learning
The authors proposed a stochastic algorithm which overcomes gradients dissimilarity using control variates as estimation of users’ variances, and by that makes FL more robust to heterogeneity in users’ data.
https://arxiv.org/abs/1910.06378

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc.

# Dataset: We use 2 datasets: FENIST, and CIFAR-10 (under construction)

- To generate Data: in main.py - function create dataset with respect to similarity betwwen clients datasets, number of clients, and name of dataset ( FMNIST, CIFAR-10 )

# Latest updates : 
We examine the performance of SCAFFOLD over noisy fading MAC and try to restore the results of the related works.
 We also examine different pre-coding scenarios of the controls.
 ![תמונה1](https://user-images.githubusercontent.com/72392859/110620699-ff6d4100-81a1-11eb-9994-459beeffad2a.png)
![תמונה2](https://user-images.githubusercontent.com/72392859/110620704-01370480-81a2-11eb-8687-f8b726de770d.png)

The figures confirm that we manage to restore related works results. In addition, it seems that when the noise applied scaffold might have degradation in performance. We suspect that controls and gradients updates tent differently over time. We use different pre-coding scaling for the controls and simulate a scenario where both pre-coding constricted to same SNR, which led to poor performance and another scenario which we allowed higher SNR in control transmission which led to desired results.
The next work is to examine how to approach to the controls pre-coding, and procced to mathematical analysis of the scheme.


-
