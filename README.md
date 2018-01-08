# ProtNets

Neural networks for protein spherical representations, based on the
[Deepfold](https://github.com/deepfold/NIPS2017) implementation by Boomsma and Frellsen.
(W. Boomsma, J. Frellsen, Spherical Convolutions and their application
in molecular modelling, Conference on Neural Information Processing
Systems (NIPS) 30, 2017.)

## Project Description

This is my project of a Convolutional Neural Network (CNN) on a spehrical
coordinate system. It tries to stay as close to state-of-the-art
CNN implementation best-practices as possible, as well as making use of
all the options available on the tensorflow framework for training,
testing and inference.

The main differences between this and DeepFold are Tensorboard usage with
summaries for training performance assessment, full tensorflow
checkpoint loading for training continuation with epoch and step as graph
variables and finally a Sparse Tensor implementation of the data loader
for faster training. Slight tweaks to the models were added as well, in
order to bring these closer to state-of-the-art code, such as batch
normalization.

## Project Organization

Part of the reasons that lead to this project was to develop my own default
project organization, as model development in tensorflow tends to be a
bit convoluted. Having a well organized project allows for easy expansion
and experimentation.

The current one is heavily influenced by
[Morgan Giraud's](https://github.com/morgangiraud) implementations,
explained [here](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3),
as well as other random discussions online about best practices.

## Sparse Tensors

Training on the dense matrix form such as the one from Deepfold data
leads to slow training times. This poor performance is mostly due to
slow transfer times between the CPU and GPU when the matrix is passed in
the dense form, leading to a lot of GPU waiting time between steps.
I've implemented a sparse matrix representation feeding mechanism
through the feed dict feature of tensorflow, leading to almost zero GPU
lag time and constant 100% usage on a single Titan X (12GB RAM) GPU.

## Future Work

This is a work-in-progress project, geared towards my own experimentation
with tensorflow and its capabilities. As such, some improvements are in
the works whenever I have the time to explore and include them, such as
queues for asynchronous data access and multi-gpu training with overall
loss calculation. Also, the code might not be the cleanest at all times. :)


