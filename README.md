# Naver Movie Sentiment Classification

## Data
All datas are from [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)

## Goal

Sentiment Classification: Classify Good/Bad from movie reviews

* Task: Many to One
* Use: Attention(to know which words is important to classify the good/bad movie) + LSTM

A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING: [https://arxiv.org/pdf/1703.03130.pdf](https://arxiv.org/pdf/1703.03130.pdf)

## Notebooks
|Date|Model|Accuracy|Link|
|:-|:-|:-|:-|
|180402|self_attn_1H_r5|0.7208|[link](https://nbviewer.jupyter.org/github/simonjisu/nsmc_study/blob/master/Notebooks/selfattn_1H_r5.ipynb)|
|180402|self_attn_1H_r20|0.8460|[link](https://nbviewer.jupyter.org/github/simonjisu/nsmc_study/blob/master/Notebooks/selfattn_1H_r20.ipynb)|
|180402|self_attn_3H_r5(over fitted)|0.8498|[link](https://nbviewer.jupyter.org/github/simonjisu/nsmc_study/blob/master/Notebooks/selfattn_3H_r5.ipynb)|

* r: n_hops of self attention
* H: number of hidden layers

## Explanation
![](/figures/model1.png)

![](/figures/model2.png)

![](/figures/model3.png)