import csv
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

with open('banknotes.csv') as f:
    reader = csv.reader(f)
    next(reader)
    
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

holdout = int(0.40 * len(data))
testing = data[:holdout]
training = data[holdout:]

X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]

model = svm.SVC()
model.fit(X_training, y_training)
predictions = model.predict(X_testing)
correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
correct = sum(correct)
incorrect = sum(incorrect)
total = len(y_testing)
print(f"Support Vector Machine:")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
print()

model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_training, y_training)
predictions = model.predict(X_testing)
correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
correct = sum(correct)
incorrect = sum(incorrect)
total = len(y_testing)
print(f"K-Nearest Neighbors (k=2):")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
print()

model = Perceptron()
model.fit(X_training, y_training)
predictions = model.predict(X_testing)
correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
correct = sum(correct)
incorrect = sum(incorrect)
total = len(y_testing)
print(f"Perceptron:")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
print()

model = GaussianNB()
model.fit(X_training, y_training)
predictions = model.predict(X_testing)
correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
correct = sum(correct)
incorrect = sum(incorrect)
total = len(y_testing)
print(f"Naive Bayes:")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")