# 

we have analyzed the convergence rate of a federated learning algorithm named SCAFFOLD (variation of SVRG) in noisy fading MAC settings and heterogenous data, in order to formulate a new algorithm that accelerates the learning process in such settings.  We inspired by 3 related works:
we depend on three articles –

1.	**On Analog Gradient Descent Learning Over Multiple Access Fading Channels
The authors implemented GBMA algorithm in which the users transmit an analog function of their local gradient using a common shaping-waveform and the network edge update the global model using a received superposition of the analog transmitted signals which represents a noisy distorted version of the gradients.
https://arxiv.org/abs/1908.07463

2.	**Over-the-Air Federated Learning from Heterogeneous Data
The authors introduce time-varying pre-coding and scaling scheme COTAF which facilitates the aggregation and gradually mitigates the noise effect and maintains the convergence properties of local SGD with heterogeneous data across users.
https://arxiv.org/abs/2009.12787

3.	**SCAFFOLD - Stochastic Controlled Averaging for Federated Learning**
>>>>>>> parent of 52069c7 (Update README.md)
The authors proposed a stochastic algorithm which overcomes gradients dissimilarity using control variates as estimation of users’ variances, and by that makes FL more robust to heterogeneity in users’ data.
https://arxiv.org/abs/1910.06378

# letest progress 
We’ve established pythonic framework that executes simulation common FedAvg, COTAF, SCAFFOLD and our proposed scheme over the extended EMNIST data in different heterogeneity scenarios. We examine the performance of SCAFFOLD over noisy fading MAC and try to restore the results of the related works. We also examine different pre-coding scenarios of the controls.
![image](https://user-images.githubusercontent.com/72392859/111066669-df52c000-84c8-11eb-9929-7391b03c04e7.png)
![image](https://user-images.githubusercontent.com/72392859/111066670-e37edd80-84c8-11eb-8ebe-cab896643205.png)

The figures confirm that we manage to restore related works results. In addition, it seems that when the noise applied scaffold might have degradation in performance. We suspect that controls and gradients updates tent differently over time. We use different pre-coding scaling for the controls and simulate a scenario where both pre-coding constricted to same SNR, which led to poor performance and another scenario which we allowed higher SNR in control transmission which led to desired results.
The next work is to examine how to approach to the controls pre-coding, and procced to mathematical analysis of the scheme


# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**


