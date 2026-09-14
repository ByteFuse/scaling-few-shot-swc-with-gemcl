---
layout: default
title: Scaling few-shot spoken classification with generative meta-continual learning
---

## Abstract
Few-shot spoken word classification has largely been developed for applications where a small number of classes is considered, and so the potential of larger-scale few-shot spoken word classification remains untapped. Our work investigates the potential of a spoken word classifier to sequentially learn to distinguish between 1000 classes when it is given only five shots per class. We demonstrate that this scaling capability exists by training a model using the Generative Meta-Continual Learning (GeMCL) algorithm and comparing it to repeatedly trained or finetuned baselines. We find that GeMCL produces exceptionally stable performance, and although it does not always outperform a repeatedly fully-finetuned HuBERT model (FT) nor a frozen HuBERT model with a repeatedly trained classifier head (CH), it produces comparable performance to the latter while adapting 2000 times faster, having been trained less than half of the data for two orders of magnitude less time.

<!-- ## Introduction
Keyword spotting (KWS) is one the applications of spoken word classification. -->

## GeMCL

Generative Meta-Continual Learning (GeMCL) [[1]](#ref1) uses a generative Bayesian classifier. GeMCL models the distribution of each class. We assume each class, $$c$$, is modelled by a Gaussian distribution with mean $$\mu^c$$ and precision $$\lambda^c$$. The class conditional Gaussians form a Gaussian Mixture Model (GMM).

The mean $$\mu^c$$ is also modelled by a Gaussian, for which we assume uninformative priors. The precision is modelled by a Gamma distribution.

The posterior distributions of the mean $$\mu^c$$  and the precision $$\lambda^c$$ take the form of a Normal-Gamma distribution. This allows us to calculate the posterior parameters using closed-form equations as we learn a class. The predictive distribution is a Student’s t-distribution.

Figure 1 illustrates the GeMCL process. Assume we are attempting to solve classification episode containing $$N$$ classes, with $$K$$ training samples for each class, which we call the support set. This is known as an $$N$$-way-$$K$$-shot episode. During an $$N$$-way-$$K$$-shot episode, the encoder receives the input and outputs the feature representation of that input. We then use Bayes' Theorem to obtain the posterior parameters of the class-specific distributions using the representation of the input and the prior.

<p align="center">
  <img src="assets/images/gemcl.png" alt="money_shot" style="max-width: 1000px; width: 100%;">
</p>
<p class="caption">Figure 1: The procedure of GeMCL in a classification episode containing 3 classes.</p>

Next, we use our model to make predictions on the query set. We use the predictive distribution to select the class that assigns the highest probability to the observed data point from the query set. 

In GeMCL, since each class possesses its own set of parameters and is modelled by separate Gaussians, updating the class-specific parameters for class $$1$$ has no effect on the parameters for classes $$2$$ and $$3$$. The class parameters are isolated. Therefore, GeMCL is immune to CF [[1]](#ref1)[[2]](#ref2).
The order in which we learn the classes does not matter. We can learn the classes in any sequence and arrive at the same distribution for that class, as the class parameters are isolated. Furthermore, GeMCL makes no assumption regarding the maximum number of classes. To learn a new class, we simply learn the statistics of that class's representations.

## Emperical Design

### Data
We make use of the English data from the Multilingual Spoken Words corpus (MSWC) dataset [[3]](#ref3). It consists of one second-long audio segments of individual words. A train/dev/test split is provided, however, we only make use of the test and train split. Each word contains samples in each split and we filter out any words that do not have at least five valid examples in the test and train splits. After the great filtering we are left with 12 736 words. The words are then randomly split into the meta-training set and meta-test set with a 70 : 30 split respectively. Therefore 8 915 words are used for meta-training and 3 821 words are used for meta-testing.

### GeMCL training
To train the encoder of GeMCL we make use meta-training. Meta-training involves training the model on a distribution of tasks (in our case $$N$$-way-$$K$$-shot episodes) whereas meta-testing is evaluating whether the model can generalise to new, unseen tasks. We make use of the meta-trained words to generate the classification episodes. We meta-train for 5 000 steps on batches of 16 25-way-5-shot episodes. The performance on the query set of the meta-training words are then used to update the encoder. 

### HuBERT baselines
We use two methods of adapting the HuBERT baseline: 1.) full finetuning 2.) training a classifier head and projector on a frozen backbone. The training runs start from the HuBERT base checkpoint.

### Evaluation

### Hyperparameters

```yaml
# GeMCL
encoder: 12-layer, 12-head transformer
mfcc:
  sample_rate: 16000
  frame_length: 25ms
  frame_shift: 10ms
  mel_filterbanks: 40
  cepstral_coeffs: 13
train_shots: 5
test_shots: 5
n_way: 25
meta_train_steps: 5000
batch_size: 16
learning_rate: 5e-5
weight_decay: 1e-2

# Full FT
learning_rate: variable (depends on number of classes)
trainable_params: all

# CH
learning_rate: 3e-4
trainable_params: projector + classifier head only

# HuBERT baselines
base_model: HuBERT base
optimizer: AdamW
batch_size: 32
epochs:
  below_300_classes: 200
  from_300_classes: 500
class_increment: 25
class_range: [25, 1000]
```

## Result
<p align="center">
  <img src="assets/images/result.png" alt="investment_shot" style="max-width: 900px; width: 100%;">
</p>
<p class="caption">Figure 2: The average accuracy of GeMCL, full FT and the CH models with 95% confidence intervals.</p>

The table shows some awesome things.

|  | GeMCL | CH | Full FT |
|---|:---:|:---:|:---:|
| Volatility (mean) | 0.48 | 7.13 | 24.55 |
| Volatility (stddev) | ±3.32 | ±13.63 | ±39.25 |

<p class="caption">Table 1: Per-word volatility of classification accuracy. Given accuracy in %, volatility is the mean absolute amount that accuracy changes between consecutive continual learning steps.</p>

|  | GeMCL | CH | Full FT |
|---|:---:|:---:|:---:|
| Meta- / Pre-train | 27.55 | ~1976 | ~1976 |
| Hyperparameter search | 15.29 | 40.26 | 70.00 |
| Few-shot adaptation | 0.06 | 124.25 | 185.96 |

<p class="caption">Table 2. Time taken in hours on a single GPU - the HuBERT numbers are calculated from 100k steps taking approximately 9.5 hours on 32 GPUs. The hours reported are cumulative - the adaptation time is taken over all shots ingested.</p>

## Conclusion
In conclusion, we conclude.

## References

1. <a id="ref1"></a>M. Banayeeanzade, R. Mirzaiezadeh, H. Hasani, M. S. Baghshah, "Generative vs Discriminative: Rethinking The Meta-Continual Learning," *Proceedings of the 35th International Conference on Neural Information Processing Systems (NeurIPS)*, pp. 21592–21604, 2021.

2. <a id="ref2"></a>S. Lee, H. Jeon, J. Son, G. Kim, "Learning to Continually Learn with the Bayesian Principle," *International Conference on Machine Learning (ICML)*, 2024.

3. <a id="ref3"></a>M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. M. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, V. Janapa Reddi, "Multilingual Spoken Words Corpus," *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. [[link]](https://openreview.net/forum?id=c20jiJ5K2H)

## Citation

```bibtex
@inproceedings{beyers2026scaling,
  title={Scaling few-shot spoken classification with generative meta-continual learning},
  author={Beyers, Louise and Ziki, Batsirayi Mupamhi and van der Merwe, Ruan},
  booktitle={Proceedings of Interspeech 2026},
  year={2026}
}
```