## Description ##

This is a port of Kevin Murphys HMM MATLAB Toolbox (http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html) to Python.

## Discussion ##

So far, there are a few implementations of HMMs for Python. There is the one in scikit-learn (http://scikit-learn.org/stable/modules/hmm.html) that has been abandoned for some time and recently resumed. However it still seems to have numerical problems and misses support for special cases of HMMs with tied parameters. Another one is the wrapper for the General Hidden Markov Model library (http://ghmm.org/), which doesn't support HMMs with Mixture of Gaussians emissions for multidimensional inputs. There are a number of smaller implementations which all lack important features or numerial stability in some part.

With the HMM toolbox, written by Kevin Murphy, there is a very good implementation of HMMs for MATLAB. To enable python users to use those algorirhms, I started a port of it to Python.

## Basic approach for the port ##

The current approach is to keep each single statements as close as possible to the original MATLAB code. This makes porting easy and helps during debugging (you can simply compare the result of each statement with the one the MATLAB version produces). However, it also introduces some inefficiencies where special properties of numpy could be used to make the code more efficient. Thus, in the future, this approach could change.

## Current State ##

So far only the Hidden Markov Model with Mixture of Gaussians Emissions has been ported and only the general case for untied parameters (mu, Sigma) has been fully tested. However, this should capture one of the most common use cases of HMMs.

Please join the project if you'd like to contribute and port parts of the library.