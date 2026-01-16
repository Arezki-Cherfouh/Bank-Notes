Banknote Authentication – Machine Learning Comparison

This project classifies banknotes as Authentic or Counterfeit using multiple supervised learning algorithms. The dataset is loaded from a CSV file and split into training and testing sets using a holdout method.

Models implemented:

Support Vector Machine (SVM)
K-Nearest Neighbors (k=2)
Perceptron
Gaussian Naive Bayes

Each model is trained on four numerical features extracted from banknotes and evaluated on unseen data. The script reports the number of correct and incorrect predictions along with accuracy for each algorithm, allowing direct performance comparison.

Technologies used:
Python, csv module, scikit-learn

Purpose:
Educational comparison of classic ML classifiers on the same dataset, focusing on simplicity, clarity, and evaluation.
