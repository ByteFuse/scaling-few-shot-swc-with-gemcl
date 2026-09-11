---
layout: default
title: Scaling few-shot spoken classification with generative meta-continual learning
---

## Abstract
Few-shot spoken word classification has largely been developed for applications where a small number of classes is considered, and so the potential of larger-scale few-shot spoken word classification remains untapped. This paper investigates the potential of a spoken word classifier to sequentially learn to distinguish between 1000 classes when it is given only five shots per class. We demonstrate that this scaling capability exists by training a model using the Generative Meta-Continual Learning (GeMCL) algorithm and comparing it to repeatedly trained or finetuned baselines. We find that GeMCL produces exceptionally stable performance, and although it does not always outperform a repeatedly fully-finetuned HuBERT model (FT) nor a frozen HuBERT model with a repeatedly trained classifier head (CH), it produces comparable performance to the latter while adapting 2000 times faster, having been trained less than half of the data for two orders of magnitude less time.

## GeMCL
<p align="center">
  <img src="assets/images/result.png" alt="money_shot" style="max-width: 900px; width: 100%;">
</p>
<p class="caption">Figure 1: The procedure of GeMCL.</p>

## Result
<p align="center">
  <img src="assets/images/gemcl.png" alt="investment_shot" style="max-width: 900px; width: 100%;">
</p>
<p class="caption">Figure 2: The average accuracy of GeMCL, full FT and the CH models with 95% confidence intervals.</p>