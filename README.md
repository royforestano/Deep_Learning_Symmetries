# Deep_Learning_Symmetries

We design a deep-learning algorithm for the discovery and identification of the continuous group of symmetries present in a labeled dataset. We use fully connected neural networks to model the symmetry transformations and the corresponding generators. We construct loss functions that ensure that the applied transformations are symmetries and that the corresponding set of generators forms a closed (sub)algebra. Our procedure is validated with several examples illustrating different types of conserved quantities preserved by symmetry. In the process of deriving the full set of symmetries, we analyze the complete subgroup structure of the rotation groups SO(2), SO(3), and SO(4), and of the Lorentz group SO(1,3). Other examples include squeeze mapping, piecewise discontinuous labels, and SO(10), demonstrating that our method is completely general, with many possible applications in physics and data science. Our study also opens the door for using a machine learning approach in the mathematical study of Lie groups and their properties.


The symmetries_tutorial.ipynb file provides a tutorial, or walkthrough, to understand the model and output functions.

The sym_demo.ipynb file, which relies on the sym_utils.py file, is a demonstration on how the plots in the paper were generated.
