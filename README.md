# Training on parameters subspaces

This project purpose is to reproduce some of the results in [Measuring the
Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
and then:

  1. report on their main properties compared to standard smaller neural network
  2. test experimentally their performance for different subspace dimensions
  3. test experimentally their performance for different families of P

To see the resuts of the experiments look at the commented lines in `runs.sh`.
The data necessary for training will be downloaded automatically with the first
run and then read from the filesystem.

A more detailed description of the project is in `report.tex`.

# Resources

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

## Sparse Random Projection

  * [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html)
  * [Very Sparse Random Projections](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf)
  * [Database-friently Random projections](http://people.ee.duke.edu/~lcarin/p93.pdf)

## FastFood transform

  * [Fastfood — Approximating Kernel Expansions in Loglinear Time](http://proceedings.mlr.press/v28/le13.pdf)
  * [Scikit-learn](https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.kernel_approximation.Fastfood.html)
  * [Scikit-learn fastfood digits](https://scikit-learn-extra.readthedocs.io/en/stable/auto_examples/kernel_approximation/plot_digits_classification_fastfood.html)
