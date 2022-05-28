Deep Learning on Graphs with Keras
====

Keras-based implementation of SSTGCN.

Abstract
------------
Accurate and real-time traffic road intersection feature extraction and similarity learning play an important role in the urban traffic control system while traditional signal timing requires a lot of manpower and time cost. We consider to measure the feature similarity between different intersections, so as to facilitate the traffic signal optimization strategy transfer of similar intersections. However, existing road intersection similarity learning methods are often distance-based measurement schemes, which are difficult to comprehensively measure spatio-temporal multivariate data along with a large amount of computation. Therefore, we propose a Siamese-Spatio-Temporal Graph Convolutional Networks (SSTGCN) with heterogeneous multi-granularity aggregation strategy to capture the underlying spatial correlations and temporal dependencies among multi-hop intersections. The experimental results show that the proposed algorithm can accurately predict the similarity of two intersections with 9.29\% lower error compared with baseline. Furthermore, it is suitable for transferring the optimized intersection traffic strategy with SSTGCN.

Installation
------------

```python setup.py install```

Dependencies
-----
  * python 3.6.13
  * numpy 1.19.2
  * keras 2.2.4
  * TensorFlow 1.12.0
  * pandas
  * scikit-learn
  * tqdm


Usage
-----

```python train.py```


