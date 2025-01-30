# Deep-Learning-Project

This repository contains an implementation of the following paper:

> [Making Transformers Solve Compositional Tasks](https://arxiv.org/abs/2108.04378), Santiago Ontanon, Joshua Ainslie, Vaclav Cvicek, Zachary Fisher, cs.AI 2022.

The code aims to replicate the first 4 tables shown in the paper.

* In the file Attention.py there is the implementation of MultiHead Attention and some variants which uses relative positional encodings.
* Code to replicate synthetic datasets is contained in Datasets.py (mostly similar to authors code).
* Encoding.py contains the implementation of the Absolute Positional Encoding module and the function to generare Relative Positional Ids.
* Transformer.py contains the standard Transformer implementation, as well as the improved architectures suggested in the paper.
* The code to train and evaluate the varius models, and so to replicate the first 4 tables, is in the file main.py.
* Finally, the 4 .csv files contains the results.

Regarding files for datasets PCFG and SCAN, links to download them are available in parer's authors github (link in paper).
