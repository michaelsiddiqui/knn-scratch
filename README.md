# knn-scratch repository

Repo to hold cold developed as part of a brush-up review and work-through
of k-nearest neighbors predictive algorithms. Will code some simple modules
with functions and/or classes to build up the functionality from base Python
libraries and then work through evaluating them on a couple tutorial-test
public datasets (Iris species dataset and Used car price dataset).

## Files, folders, and structure

Currently Jupyter notebooks and Python modules are in the root directory. These
files include:

* This README markdown file
* knn_base Python module
* 1_Base_function_development notebook
* 2_Plot_iris_dataset notebook
* 3_Score_iris_algorithm notebook

A `tests` subfolder includes testing modules with unittest.TestCase classes
written with methods to test the functions in the base module

A `data` subfolder includes the two dataset files in `csv` format

* iris_data.csv
* usedcars.csv

## Other notes

Jupyter notebooks include markdown text describing the purpose and structure
of the coded workflows executed in the interactive sessions

Python modules have docstrings intended to give reader/developer context on
usage.

#### How to optimize distance weightings in a regression algorithm

*x<sup>(i)</sup><sub>j</sub>* = value of feature *j* in the *i*<sup>th</sup> training example, in knn regression the features are the distance scores

*x<sup>(i)</sup>* = the input (features) of the *i*<sup>th</sup> training example

*m* = the number of training examples

*n* = the number of features

The prediction function, h<sub>&theta;</sub>(X):

- h<sub>&theta;</sub>(X) = &theta;<sup>T</sup>x = &Sigma; <sub>j=0</sub><sup>n</sup> ( &theta;<sub>j</sup>\*x<sub>j</sub> )

Cost function, J(&Theta;):

  - J(&theta;) = &Sigma;<sub>i=0</sub><sup>m</sup> ( h<sub>&theta;</sub>(x<sup>i</sup>) - y<sup>i</sup> )<sup>2</sup>



#### TODOs:

* Cleanup root directory to reduce file clutter
* Rewrite "on the fly" function definitions in Jupyter notebooks to modules
* Write additional test methods for increased function coverage
* Define class to wrap inter-related function definitions as methods
* Rewrite initial knn_base functions to use numpy array objects
