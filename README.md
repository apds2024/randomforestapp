# A Random Forest Walk

This Streamlit app uses a *RandomForestRegressor()* algorithm to build a regression model for the Ames housing dataset. The user can adjust the hyperparameters to see how they affect the results.

The RandomForestRegressor is an ensemble learning method that builds multiple decision trees
from random subsets of the training data and averages their predictions to improve accuracy and reduce overfitting.
It uses bootstrapping (sampling with replacement) and random feature selection to create diverse trees. 

This approach results in a more robust and accurate prediction model compared to individual decision trees.
