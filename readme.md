# SSL Encoder Shareability Metric

A theoretical and empirical framework for measuring when two self-supervised learning views can effectively share a common encoder.

**[Read the full paper](paper/SSL_Encoder_Sharability_Metric.pdf)**

## Overview

This project introduces a shareability metric for quantifying how much performance is retained when using a shared encoder instead of separate encoders. The metric is designed to indiactes when shared encoder is likeley to be sufficient without requireing both configurations to be fully trained and evaluated, potentially reducing computatinoal cost and training time. The current work is restricted to the linear setting, which allows for exact theoretical analysis while future work will investigate extensions to nonlinear encoders. 

## Key Idea

The shareability metric is defined as

$$
p(M) = \frac{J_{shared}^*}{J_{sep}^*} = \frac{max_i|\lambda_i(S)|}{\sigma(M)} \qquad M \neq 0,
$$

where $M=\mathcal{E}|\tilde{y}\tilede{x}^\top|$ denotes the cross-covairnace matrix between whitened and centered $x$ and $y$ and $S=\frac{M + M^\top}{2}$ denotes the symetric part of $M$.

## Results