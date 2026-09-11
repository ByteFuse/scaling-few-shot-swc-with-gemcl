---
layout: default
title: Scaling few-shot spoken classification with generative meta-continual learning
---

## Abstract
Few-shot spoken word classification has largely been developed for applications where a small number of classes is considered, and so the potential of larger-scale few-shot spoken word classification remains untapped. This paper investigates the potential of a spoken word classifier to sequentially learn to distinguish between 1000 classes when it is given only five shots per class. We demonstrate that this scaling capability exists by training a model using the Generative Meta-Continual Learning (GeMCL) algorithm and comparing it to repeatedly trained or finetuned baselines. We find that GeMCL produces exceptionally stable performance, and although it does not always outperform a repeatedly fully-finetuned HuBERT model (FT) nor a frozen HuBERT model with a repeatedly trained classifier head (CH), it produces comparable performance to the latter while adapting 2000 times faster, having been trained less than half of the data for two orders of magnitude less time.

## GeMCL

Generative Meta-Continual Learning (GeMCL) [[1]](#ref1) uses a generative Bayesian classifier. GeMCL models the distribution of each class. We assume each class, $$c$$, is modelled by a Gaussian distribution with mean $$\mu^c$$ and precision $$\lambda^c$$. The class conditional Gaussians form a Gaussian Mixture Model (GMM).

The mean $$\mu^c$$ is also modelled by a Gaussian, for which we assume uninformative priors. The precision is modelled by a Gamma distribution.

The posterior distributions of the mean $$\mu^c$$  and the precision $$\lambda^c$$ take the form of a Normal-Gamma distribution. This allows us to calculate the posterior parameters using closed-form equations as we learn a class. The predictive distribution is a Student’s t-distribution.

Figure 1 illustrates the GeMCL process. During an $$N$$-way-$$K$$-shot episode, the encoder receives the input and outputs the feature representation of that input. We then use Bayes' Theorem to obtain the posterior parameters of the class-specific distributions using the representation of the input and the prior. The posterior parameters then act as the prior for the next step. This process continues until we have iterated through the entire support set.

Next, we use our model to make predictions on the query set. The encoder and the prior parameters are fixed during this step. We use the predictive distribution to select the class that assigns the highest probability to the observed data point from the query set. This loss is then used to update the encoder.

In GeMCL, since each class possesses its own set of parameters and is modelled by separate Gaussians, updating the class-specific parameters for class $$1$$ has no effect on the parameters for classes $$2$$ and $$3$$. The class parameters are isolated. Therefore, GeMCL is immune to CF [[1]](#ref1)[[2]](#ref2).
The order in which we learn the classes does not matter. We can learn the classes in any sequence and arrive at the same distribution for that class, as the class parameters are isolated. Furthermore, GeMCL makes no assumption regarding the maximum number of classes. To learn a new class, we simply learn the statistics of that class's representations.

<p align="center">
  <img src="assets/images/gemcl.png" alt="money_shot" style="max-width: 900px; width: 100%;">
</p>
<p class="caption">Figure 1: The procedure of GeMCL.</p>

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

## References

1. <a id="ref1"></a>M. Banayeeanzade, R. Mirzaiezadeh, H. Hasani, M. S. Baghshah, "Generative vs Discriminative: Rethinking The Meta-Continual Learning," *Proceedings of the 35th International Conference on Neural Information Processing Systems (NeurIPS)*, pp. 21592–21604, 2021.

2. <a id="ref2"></a>S. Lee, H. Jeon, J. Son, G. Kim, "Learning to Continually Learn with the Bayesian Principle," *International Conference on Machine Learning (ICML)*, 2024.

## Citation

```bibtex
@inproceedings{beyers2026scaling,
  title={Scaling few-shot spoken classification with generative meta-continual learning},
  author={Beyers, Louise and Ziki, Batsirayi Mupamhi and van der Merwe, Ruan},
  booktitle={Proceedings of Interspeech 2026},
  year={2026}
}
```