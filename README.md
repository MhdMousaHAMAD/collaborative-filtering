# About
This code is a simple implementation for item-based and user-based collaborative filtering methods which make automatic predictions about the interset of a user with an item. The interest is expressed as a rating value between 1 and 5.

This code also provides a function to evaluate the implemented IBCF and UBCF using cross-validataion based on the Root Mean Square Error (RMSE).

This implementation is based mainly on the descritption and codes provided in the book "Programming Collective Intelligence: Building Smart Web 2.0 Applications" by Toby Segaran.

Both IBCF and UBCF are implemented in the same code with a flag in every function specific to each of the methods that tells whether you are calling IBCF or UBCF.

# Usage
To traing a model, the code needs a training set of previous predictions represented in a CSV file that is formatted as (user, item, rating) triples.
See "training.dat" file in "data" directory for an example of a training dataset.

To predict how a user would rate an item, you can provide the IDs for both a user and an item to make a single prediction. You can also predict missing ratings for a complete set represented in a CSV file that is formatted as (user, item) pairs.
    
Some extra wrapper functions are implemented in this code to provide different variations for calling methods.

A simple code to train, cross-validate and use a collaborative filtering model is provided in the "main.py" file.
