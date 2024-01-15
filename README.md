# Project 03: Training on parameters subspaces

Stuff from the authors:
  * [arXiv](https://arxiv.org/abs/1804.08838)
  * [GitHub](https://github.com/uber-research/intrinsic-dimension)
  * [Uber Engineering](https://eng.uber.com/intrinsic-dimension/)
  * [YouTube](https://www.youtube.com/watch?v=uSZWeRADTFI)
  * [Papers with Code](https://paperswithcode.com/paper/measuring-the-intrinsic-dimension-of)

Stuff from other people:
  * [VITALab](https://vitalab.github.io/article/2018/05/11/intrinsic-dimension-objective-landscapes.html) excellent explanation!
  * [A good set of slides](https://www.cs.princeton.edu/~runzhey/demo/Geo-Intrinsic-Dimension.pdf) from one of the authors.
  * [A study](https://aclanthology.org/2021.acl-long.568.pdf) using intrinsic dimensionality.
  * [SlideShare](https://www.slideshare.net/WuhyunRicoShin/paper-review-measuring-the-intrinsic-dimension-of-objective-landscapes)
  * A non particularly revealing [article](https://tomroth.com.au/notes/intdim/intdim/)

# Designing the experiments

I was asked to:
  1. report on their main properties compared to standard smaller neural network
  2. test experimentally their performance for different subspace dimensions
  3. test experimentally their performance for different families of P
Points 2 and 3 are basically the same as reproducing the results of the original
paper. For point 1 a good comparison may be between a neural network with the 
same number of parameters as the intrinsic dimension of another.

They in the paper did:
  * FC and ConvNet network on MNIST
  * FC and ConvNet network on CIFAR-10
  * Some stuff on reinforcement learning
  * Used the Fastfood transform.

Multivariate data (i.e. data in tabular format as in one row per observation) is
ideal for feed forward neural network.

The convolutional network I'll implement is LeNet.

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

https://youtube.com/watch?v=O2wJ3tkc-TU common mistakes

# Sparse Random Projection
https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html

https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf
http://people.ee.duke.edu/~lcarin/p93.pdf

# FastFood transform
http://proceedings.mlr.press/v28/le13.pdf

https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.kernel_approximation.Fastfood.html
https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/kernel_approximation/plot_digits_classification_fastfood.html
