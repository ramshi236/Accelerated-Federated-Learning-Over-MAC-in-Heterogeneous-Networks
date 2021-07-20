# **Accelerated Federated Learning Over MAC in Heterogeneous Networks**

I have analyzed the convergence rate of a federated learning algorithm named SCAFFOLD (variation of SVRG) in noisy fading MAC settings and heterogenous data, in order to formulate a new algorithm that accelerates the learning process in such settings.  I inspired by 3 related works:
I depend on three articles –

1.	**On Analog Gradient Descent Learning Over Multiple Access Fading Channels**
The authors implemented GBMA algorithm in which the users transmit an analog function of their local gradient using a common shaping-waveform and the network edge update the global model using a received superposition of the analog transmitted signals which represents a noisy distorted version of the gradients.
https://arxiv.org/abs/1908.07463

2.	**Over-the-Air Federated Learning from Heterogeneous Data**
The authors introduce time-varying pre-coding and scaling scheme COTAF which facilitates the aggregation and gradually mitigates the noise effect and maintains the convergence properties of local SGD with heterogeneous data across users.
https://arxiv.org/abs/2009.12787

3.	**SCAFFOLD - Stochastic Controlled Averaging for Federated Learning**
The authors proposed a stochastic algorithm which overcomes gradients dissimilarity using control variates as estimation of users’ variances, and by that makes FL more robust to heterogeneity in users’ data.
https://arxiv.org/abs/1910.06378

# letest progress 
I’ve established pythonic framework that executes simulation common FedAvg, COTAF, SCAFFOLD and our proposed scheme over the extended EMNIST data in different heterogeneity scenarios. I examine the performance of SCAFFOLD over noisy fading MAC and try to restore the results of the related works. I also examine different pre-coding scenarios of the controls.

![WhatsApp Image 2021-03-09 at 16 37 08](https://user-images.githubusercontent.com/72392859/111066827-b4b53700-84c9-11eb-8b5c-f9d1dd01ff7e.jpeg)

We analyzed the model and control update norms during the learning process. The model and the control norms have a constant proportion. Conclusion :try to apply different precoding to each parameter type.  

![image](https://user-images.githubusercontent.com/72392859/126355502-10650454-27c4-47f1-a73c-45344c46b10c.png)


![WhatsApp Image 2021-03-09 at 16 34 31](https://user-images.githubusercontent.com/72392859/111066830-b7b02780-84c9-11eb-8f69-152bc0f83969.jpeg)



The figures confirm that I manage to restore related works results. In addition, it seems that when the noise applied scaffold might have degradation in performance. We suspected that controls and gradients updates tent differently over time. we use different pre-coding scaling for the controls and simulate a scenario where both pre-coding constricted to same SNR

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**


