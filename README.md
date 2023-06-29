# Deep_Learning_Symmetries

---

Connected to the paper on: Deep Learning Symmetries and Their Lie Groups, Algebras, and Subalgebras from First Principles 
Published in Mach. Learn. Sci. Tech. v. 4, no.2, 025027 (2023) doi:[10.1088/2632-2153/acd989]
[https://iopscience.iop.org/article/10.1088/2632-2153/acd989]. 
(arXiv:2301.05638: [https://arxiv.org/abs/2301.05638]).

We design a deep-learning algorithm for the discovery and identification of the continuous group of symmetries present in a labeled dataset. We use fully connected neural networks to model the symmetry transformations and the corresponding generators. We construct loss functions that ensure that the applied transformations are symmetries and that the corresponding set of generators forms a closed (sub)algebra. Our procedure is validated with several examples illustrating different types of conserved quantities preserved by symmetry. In the process of deriving the full set of symmetries, we analyze the complete subgroup structure of the rotation groups SO(2), SO(3), and SO(4), and of the Lorentz group SO(1,3). Other examples include squeeze mapping, piecewise discontinuous labels, and SO(10), demonstrating that our method is completely general, with many possible applications in physics and data science. Our study also opens the door for using a machine learning approach in the mathematical study of Lie groups and their properties.


The symmetries_tutorial.ipynb file provides a tutorial, or walkthrough, to understand the model and output functions.

The sym_demo.ipynb file, which relies on the sym_utils.py file, is a demonstration on how the plots in the paper were generated.

---

Connected to the paper on: Discovering Sparse Representations of Lie Groups with Machine Learning (arXiv:2302.05383: [https://arxiv.org/abs/2302.05383]).

Recent work has used deep learning to derive symmetry transformations, which preserve conserved quantities, and to obtain the corresponding algebras of generators. In this letter, we extend this technique to derive sparse representations of arbitrary Lie algebras. We show that our method reproduces the canonical (sparse) representations of the generators of the Lorentz group, as well as the U(n) and SU(n) families of Lie groups. This approach is completely general and can be used to find the infinitesimal generators for any Lie group.

The sym_u_and_su_tutorial.ipynb file provides a tutorial, or walkthrough, to understand the model and output functions.

The sym_u_and_su_demo.ipynb file, which relies on the sym_u_and_su.py file, is a demonstration on how the plots in the paper were generated.


---

Connected to the paper on: Oracle-Preserving Latent Flows (arXiv:2302.00806:[https://arxiv.org/abs/2302.00806]).

We develop a deep learning methodology for the simultaneous discovery of multiple nontrivial continuous symmetries across an entire labelled dataset. The symmetry transformations and the corresponding generators are modeled with fully connected neural networks trained with a specially constructed loss function ensuring the desired symmetry properties. The two new elements in this work are the use of a reduced-dimensionality latent space and the generalization to transformations invariant with respect to high-dimensional oracles. The method is demonstrated with several examples on the MNIST digit dataset.


