import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import seaborn as sns
import joblib
import os

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

# Read data
with open('banknotes.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Split data
holdout = int(0.40 * len(data))
testing = data[:holdout]
training = data[holdout:]

X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]

# Create numeric versions of labels for gradient boosting models
y_training_numeric = [0 if label == "Authentic" else 1 for label in y_training]
y_testing_numeric = [0 if label == "Authentic" else 1 for label in y_testing]

# Store results
models = []
results = []

# Define models to test
model_configs = [
    ("Support Vector Machine", svm.SVC(), False),
    ("K-Nearest Neighbors (k=2)", KNeighborsClassifier(n_neighbors=2), False),
    ("Perceptron", Perceptron(), False),
    ("Naive Bayes", GaussianNB(), False),
    ("Logistic Regression", LogisticRegression(max_iter=1000), False),
    ("Linear Regression", LinearRegression(), True),
    ("Gradient Boosting Machine", GradientBoostingClassifier(), False),
    ("XGBoost", xgb.XGBClassifier(eval_metric='logloss'), True),
    ("LightGBM", lgb.LGBMClassifier(verbose=-1), True),
    ("CatBoost", CatBoostClassifier(verbose=0), True)
]

# Train and evaluate each model
for model_name, model, needs_numeric in model_configs:
    # Use numeric labels for gradient boosting models that require them
    y_train = y_training_numeric if needs_numeric else y_training
    y_test = y_testing_numeric if needs_numeric else y_testing
    
    model.fit(X_training, y_train)
    predictions = model.predict(X_testing)
    
    # Save model weights
    model_filename = f"saved_models/{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.joblib"
    joblib.dump(model, model_filename)
    print(f"Saved model: {model_filename}")
    
    # Convert predictions back to string labels for comparison if needed
    if needs_numeric:
        predictions = ["Authentic" if p == 0 else "Counterfeit" for p in predictions]
    
    correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
    incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
    total = len(y_testing)
    accuracy = 100 * correct / total
    
    print(f"{model_name}:")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    
    models.append(model_name)
    results.append({
        'name': model_name,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'predictions': predictions,
        'model': model
    })

# Example: Load a saved model
# loaded_model = joblib.load('saved_models/Support_Vector_Machine.joblib')
# predictions = loaded_model.predict(X_testing)

# Visualization 1: Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
accuracies = [r['accuracy'] for r in results]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, 105)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: accuracy_comparison.png")
plt.close()

# Visualization 2: Correct vs Incorrect Predictions
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

correct_counts = [r['correct'] for r in results]
incorrect_counts = [r['incorrect'] for r in results]

bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', 
               color='#27ae60', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
               color='#e74c3c', edgecolor='black', linewidth=1.2)

ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title('Correct vs Incorrect Predictions by Model', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
print("Saved: correct_vs_incorrect.png")
plt.close()

# Visualization 3: Confusion Matrices for all models
fig, axes = plt.subplots(4, 3, figsize=(16, 18))
axes = axes.ravel()

for idx, result in enumerate(results):
    cm = confusion_matrix(y_testing, result['predictions'], 
                          labels=["Authentic", "Counterfeit"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=["Authentic", "Counterfeit"])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2f}%", 
                       fontweight='bold', fontsize=11)
    axes[idx].grid(False)

# Hide unused subplots
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")
plt.close()

# Visualization 4: Feature Distribution (first two features)
plt.figure(figsize=(12, 8))

X_training_array = np.array(X_training)
X_testing_array = np.array(X_testing)

# Plot training data
authentic_train = [X_training_array[i] for i in range(len(X_training_array)) 
                   if y_training[i] == "Authentic"]
counterfeit_train = [X_training_array[i] for i in range(len(X_training_array)) 
                     if y_training[i] == "Counterfeit"]

if authentic_train:
    authentic_train = np.array(authentic_train)
    plt.scatter(authentic_train[:, 0], authentic_train[:, 1], 
               c='green', marker='o', s=50, alpha=0.6, 
               edgecolors='black', linewidth=0.5, label='Authentic (Train)')

if counterfeit_train:
    counterfeit_train = np.array(counterfeit_train)
    plt.scatter(counterfeit_train[:, 0], counterfeit_train[:, 1], 
               c='red', marker='s', s=50, alpha=0.6, 
               edgecolors='black', linewidth=0.5, label='Counterfeit (Train)')

plt.xlabel('Feature 1 (Variance)', fontsize=12, fontweight='bold')
plt.ylabel('Feature 2 (Skewness)', fontsize=12, fontweight='bold')
plt.title('Banknote Feature Distribution (Training Data)', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: feature_distribution.png")
plt.close()

# Visualization 5: Model Performance Summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Accuracy with error indication
accuracies = [r['accuracy'] for r in results]
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

bars = ax1.barh(models, accuracies, color=colors_gradient, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlim(0, 105)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

# Right plot: Pie chart of best model performance
best_result = max(results, key=lambda x: x['accuracy'])
sizes = [best_result['correct'], best_result['incorrect']]
labels = ['Correct', 'Incorrect']
colors_pie = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title(f'Best Model: {best_result["name"]}\nAccuracy: {best_result["accuracy"]:.2f}%', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
print("Saved: performance_summary.png")
plt.close()

print("\n✓ All visualizations saved successfully!")
print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")


# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import seaborn as sns
# import joblib
# import os

# # Create directory for saved models
# os.makedirs('saved_models', exist_ok=True)

# # Read data
# with open('banknotes.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     data = []
#     for row in reader:
#         data.append({
#             "evidence": [float(cell) for cell in row[:4]],
#             "label": "Authentic" if row[4] == "0" else "Counterfeit"
#         })

# # Split data
# holdout = int(0.40 * len(data))
# testing = data[:holdout]
# training = data[holdout:]

# X_training = [row["evidence"] for row in training]
# y_training = [row["label"] for row in training]
# X_testing = [row["evidence"] for row in testing]
# y_testing = [row["label"] for row in testing]

# # Create numeric versions of labels for gradient boosting models
# y_training_numeric = [0 if label == "Authentic" else 1 for label in y_training]
# y_testing_numeric = [0 if label == "Authentic" else 1 for label in y_testing]

# # Store results
# models = []
# results = []

# # Define models to test
# model_configs = [
#     ("Support Vector Machine", svm.SVC(), False),
#     ("K-Nearest Neighbors (k=2)", KNeighborsClassifier(n_neighbors=2), False),
#     ("Perceptron", Perceptron(), False),
#     ("Naive Bayes", GaussianNB(), False),
#     ("Logistic Regression", LogisticRegression(max_iter=1000), False),
#     ("Linear Regression", LinearRegression(), True),
#     ("Gradient Boosting Machine", GradientBoostingClassifier(), False),
#     ("XGBoost", xgb.XGBClassifier(eval_metric='logloss'), True),
#     ("LightGBM", lgb.LGBMClassifier(verbose=-1), True),
#     ("CatBoost", CatBoostClassifier(verbose=0), True)
# ]

# # Train and evaluate each model
# for model_name, model, needs_numeric in model_configs:
#     # Use numeric labels for gradient boosting models that require them
#     y_train = y_training_numeric if needs_numeric else y_training
#     y_test = y_testing_numeric if needs_numeric else y_testing
    
#     model.fit(X_training, y_train)
#     predictions = model.predict(X_testing)
    
#     # Save model weights
#     model_filename = f"saved_models/{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.pkl"
#     joblib.dump(model, model_filename)
#     print(f"Saved model: {model_filename}")
    
#     # Convert predictions back to string labels for comparison if needed
#     if needs_numeric:
#         predictions = ["Authentic" if p == 0 else "Counterfeit" for p in predictions]
    
#     correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
#     incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
#     total = len(y_testing)
#     accuracy = 100 * correct / total
    
#     print(f"{model_name}:")
#     print(f"Correct: {correct}")
#     print(f"Incorrect: {incorrect}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print()
    
#     models.append(model_name)
#     results.append({
#         'name': model_name,
#         'correct': correct,
#         'incorrect': incorrect,
#         'accuracy': accuracy,
#         'predictions': predictions,
#         'model': model
#     })

# # Example: Load a saved model
# # loaded_model = joblib.load('saved_models/Support_Vector_Machine.pkl')
# # predictions = loaded_model.predict(X_testing)

# # Visualization 1: Accuracy Comparison Bar Chart
# plt.figure(figsize=(10, 6))
# accuracies = [r['accuracy'] for r in results]
# colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
# bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
# plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# plt.xlabel('Model', fontsize=12, fontweight='bold')
# plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
# plt.ylim(0, 105)
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on bars
# for bar, acc in zip(bars, accuracies):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#              f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# plt.tight_layout()
# plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
# print("Saved: accuracy_comparison.png")
# plt.close()

# # Visualization 2: Correct vs Incorrect Predictions
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.arange(len(models))
# width = 0.35

# correct_counts = [r['correct'] for r in results]
# incorrect_counts = [r['incorrect'] for r in results]

# bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', 
#                color='#27ae60', edgecolor='black', linewidth=1.2)
# bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
#                color='#e74c3c', edgecolor='black', linewidth=1.2)

# ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
# ax.set_xlabel('Model', fontsize=12, fontweight='bold')
# ax.set_title('Correct vs Incorrect Predictions by Model', fontsize=14, fontweight='bold', pad=20)
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45, ha='right')
# ax.legend(fontsize=11)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 2,
#                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.savefig('correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
# print("Saved: correct_vs_incorrect.png")
# plt.close()

# # Visualization 3: Confusion Matrices for all models
# fig, axes = plt.subplots(4, 3, figsize=(16, 18))
# axes = axes.ravel()

# for idx, result in enumerate(results):
#     cm = confusion_matrix(y_testing, result['predictions'], 
#                           labels=["Authentic", "Counterfeit"])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                                    display_labels=["Authentic", "Counterfeit"])
#     disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
#     axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2f}%", 
#                        fontweight='bold', fontsize=11)
#     axes[idx].grid(False)

# # Hide unused subplots
# for idx in range(len(results), len(axes)):
#     axes[idx].axis('off')

# plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=0.995)
# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
# print("Saved: confusion_matrices.png")
# plt.close()

# # Visualization 4: Feature Distribution (first two features)
# plt.figure(figsize=(12, 8))

# X_training_array = np.array(X_training)
# X_testing_array = np.array(X_testing)

# # Plot training data
# authentic_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                    if y_training[i] == "Authentic"]
# counterfeit_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                      if y_training[i] == "Counterfeit"]

# if authentic_train:
#     authentic_train = np.array(authentic_train)
#     plt.scatter(authentic_train[:, 0], authentic_train[:, 1], 
#                c='green', marker='o', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Authentic (Train)')

# if counterfeit_train:
#     counterfeit_train = np.array(counterfeit_train)
#     plt.scatter(counterfeit_train[:, 0], counterfeit_train[:, 1], 
#                c='red', marker='s', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Counterfeit (Train)')

# plt.xlabel('Feature 1 (Variance)', fontsize=12, fontweight='bold')
# plt.ylabel('Feature 2 (Skewness)', fontsize=12, fontweight='bold')
# plt.title('Banknote Feature Distribution (Training Data)', fontsize=14, fontweight='bold', pad=20)
# plt.legend(fontsize=11)
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
# print("Saved: feature_distribution.png")
# plt.close()

# # Visualization 5: Model Performance Summary
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left plot: Accuracy with error indication
# accuracies = [r['accuracy'] for r in results]
# colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

# bars = ax1.barh(models, accuracies, color=colors_gradient, edgecolor='black', linewidth=1.2)
# ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold', pad=20)
# ax1.set_xlim(0, 105)
# ax1.grid(axis='x', alpha=0.3, linestyle='--')

# for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
#     width = bar.get_width()
#     ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
#              f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

# # Right plot: Pie chart of best model performance
# best_result = max(results, key=lambda x: x['accuracy'])
# sizes = [best_result['correct'], best_result['incorrect']]
# labels = ['Correct', 'Incorrect']
# colors_pie = ['#2ecc71', '#e74c3c']
# explode = (0.05, 0)

# ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
#         autopct='%1.1f%%', shadow=True, startangle=90,
#         textprops={'fontsize': 12, 'fontweight': 'bold'})
# ax2.set_title(f'Best Model: {best_result["name"]}\nAccuracy: {best_result["accuracy"]:.2f}%', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
# print("Saved: performance_summary.png")
# plt.close()

# print("\n✓ All visualizations saved successfully!")
# print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")










# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import seaborn as sns

# # Read data
# with open('banknotes.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     data = []
#     for row in reader:
#         data.append({
#             "evidence": [float(cell) for cell in row[:4]],
#             "label": "Authentic" if row[4] == "0" else "Counterfeit"
#         })

# # Split data
# holdout = int(0.40 * len(data))
# testing = data[:holdout]
# training = data[holdout:]

# X_training = [row["evidence"] for row in training]
# y_training = [row["label"] for row in training]
# X_testing = [row["evidence"] for row in testing]
# y_testing = [row["label"] for row in testing]

# # Create numeric versions of labels for gradient boosting models
# y_training_numeric = [0 if label == "Authentic" else 1 for label in y_training]
# y_testing_numeric = [0 if label == "Authentic" else 1 for label in y_testing]

# # Store results
# models = []
# results = []

# # Define models to test
# model_configs = [
#     ("Support Vector Machine", svm.SVC(), False),
#     ("K-Nearest Neighbors (k=2)", KNeighborsClassifier(n_neighbors=2), False),
#     ("Perceptron", Perceptron(), False),
#     ("Naive Bayes", GaussianNB(), False),
#     ("Logistic Regression", LogisticRegression(max_iter=1000), False),
#     ("Linear Regression", LinearRegression(), True),
#     ("Gradient Boosting Machine", GradientBoostingClassifier(), False),
#     ("XGBoost", xgb.XGBClassifier(eval_metric='logloss'), True),
#     ("LightGBM", lgb.LGBMClassifier(verbose=-1), True),
#     ("CatBoost", CatBoostClassifier(verbose=0), True)
# ]

# # Train and evaluate each model
# for model_name, model, needs_numeric in model_configs:
#     # Use numeric labels for gradient boosting models that require them
#     y_train = y_training_numeric if needs_numeric else y_training
#     y_test = y_testing_numeric if needs_numeric else y_testing
    
#     model.fit(X_training, y_train)
#     predictions = model.predict(X_testing)
    
#     # Convert predictions back to string labels for comparison if needed
#     if needs_numeric:
#         predictions = ["Authentic" if p == 0 else "Counterfeit" for p in predictions]
    
#     correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
#     incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
#     total = len(y_testing)
#     accuracy = 100 * correct / total
    
#     print(f"{model_name}:")
#     print(f"Correct: {correct}")
#     print(f"Incorrect: {incorrect}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print()
    
#     models.append(model_name)
#     results.append({
#         'name': model_name,
#         'correct': correct,
#         'incorrect': incorrect,
#         'accuracy': accuracy,
#         'predictions': predictions,
#         'model': model
#     })

# # Visualization 1: Accuracy Comparison Bar Chart
# plt.figure(figsize=(10, 6))
# accuracies = [r['accuracy'] for r in results]
# colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
# bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
# plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# plt.xlabel('Model', fontsize=12, fontweight='bold')
# plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
# plt.ylim(0, 105)
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on bars
# for bar, acc in zip(bars, accuracies):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#              f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# plt.tight_layout()
# plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
# print("Saved: accuracy_comparison.png")
# plt.close()

# # Visualization 2: Correct vs Incorrect Predictions
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.arange(len(models))
# width = 0.35

# correct_counts = [r['correct'] for r in results]
# incorrect_counts = [r['incorrect'] for r in results]

# bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', 
#                color='#27ae60', edgecolor='black', linewidth=1.2)
# bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
#                color='#e74c3c', edgecolor='black', linewidth=1.2)

# ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
# ax.set_xlabel('Model', fontsize=12, fontweight='bold')
# ax.set_title('Correct vs Incorrect Predictions by Model', fontsize=14, fontweight='bold', pad=20)
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45, ha='right')
# ax.legend(fontsize=11)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 2,
#                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.savefig('correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
# print("Saved: correct_vs_incorrect.png")
# plt.close()

# # Visualization 3: Confusion Matrices for all models
# fig, axes = plt.subplots(4, 3, figsize=(16, 18))
# axes = axes.ravel()

# for idx, result in enumerate(results):
#     cm = confusion_matrix(y_testing, result['predictions'], 
#                           labels=["Authentic", "Counterfeit"])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                                    display_labels=["Authentic", "Counterfeit"])
#     disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
#     axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2f}%", 
#                        fontweight='bold', fontsize=11)
#     axes[idx].grid(False)

# # Hide unused subplots
# for idx in range(len(results), len(axes)):
#     axes[idx].axis('off')

# plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=0.995)
# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
# print("Saved: confusion_matrices.png")
# plt.close()

# # Visualization 4: Feature Distribution (first two features)
# plt.figure(figsize=(12, 8))

# X_training_array = np.array(X_training)
# X_testing_array = np.array(X_testing)

# # Plot training data
# authentic_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                    if y_training[i] == "Authentic"]
# counterfeit_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                      if y_training[i] == "Counterfeit"]

# if authentic_train:
#     authentic_train = np.array(authentic_train)
#     plt.scatter(authentic_train[:, 0], authentic_train[:, 1], 
#                c='green', marker='o', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Authentic (Train)')

# if counterfeit_train:
#     counterfeit_train = np.array(counterfeit_train)
#     plt.scatter(counterfeit_train[:, 0], counterfeit_train[:, 1], 
#                c='red', marker='s', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Counterfeit (Train)')

# plt.xlabel('Feature 1 (Variance)', fontsize=12, fontweight='bold')
# plt.ylabel('Feature 2 (Skewness)', fontsize=12, fontweight='bold')
# plt.title('Banknote Feature Distribution (Training Data)', fontsize=14, fontweight='bold', pad=20)
# plt.legend(fontsize=11)
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
# print("Saved: feature_distribution.png")
# plt.close()

# # Visualization 5: Model Performance Summary
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left plot: Accuracy with error indication
# accuracies = [r['accuracy'] for r in results]
# colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

# bars = ax1.barh(models, accuracies, color=colors_gradient, edgecolor='black', linewidth=1.2)
# ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold', pad=20)
# ax1.set_xlim(0, 105)
# ax1.grid(axis='x', alpha=0.3, linestyle='--')

# for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
#     width = bar.get_width()
#     ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
#              f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

# # Right plot: Pie chart of best model performance
# best_result = max(results, key=lambda x: x['accuracy'])
# sizes = [best_result['correct'], best_result['incorrect']]
# labels = ['Correct', 'Incorrect']
# colors_pie = ['#2ecc71', '#e74c3c']
# explode = (0.05, 0)

# ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
#         autopct='%1.1f%%', shadow=True, startangle=90,
#         textprops={'fontsize': 12, 'fontweight': 'bold'})
# ax2.set_title(f'Best Model: {best_result["name"]}\nAccuracy: {best_result["accuracy"]:.2f}%', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
# print("Saved: performance_summary.png")
# plt.close()

# print("\n✓ All visualizations saved successfully!")
# print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")




# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.linear_model import Perceptron
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import seaborn as sns

# # Read data
# with open('banknotes.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     data = []
#     for row in reader:
#         data.append({
#             "evidence": [float(cell) for cell in row[:4]],
#             "label": "Authentic" if row[4] == "0" else "Counterfeit"
#         })

# # Split data
# holdout = int(0.40 * len(data))
# testing = data[:holdout]
# training = data[holdout:]

# X_training = [row["evidence"] for row in training]
# y_training = [row["label"] for row in training]
# X_testing = [row["evidence"] for row in testing]
# y_testing = [row["label"] for row in testing]

# # Create numeric versions of labels for gradient boosting models
# y_training_numeric = [0 if label == "Authentic" else 1 for label in y_training]
# y_testing_numeric = [0 if label == "Authentic" else 1 for label in y_testing]

# # Store results
# models = []
# results = []

# # Define models to test
# model_configs = [
#     ("Support Vector Machine", svm.SVC(), False),
#     ("K-Nearest Neighbors (k=2)", KNeighborsClassifier(n_neighbors=2), False),
#     ("Perceptron", Perceptron(), False),
#     ("Naive Bayes", GaussianNB(), False),
#     ("Gradient Boosting Machine", GradientBoostingClassifier(), False),
#     ("XGBoost", xgb.XGBClassifier(eval_metric='logloss'), True),
#     ("LightGBM", lgb.LGBMClassifier(verbose=-1), True),
#     ("CatBoost", CatBoostClassifier(verbose=0), True)
# ]

# # Train and evaluate each model
# for model_name, model, needs_numeric in model_configs:
#     # Use numeric labels for gradient boosting models that require them
#     y_train = y_training_numeric if needs_numeric else y_training
#     y_test = y_testing_numeric if needs_numeric else y_testing
    
#     model.fit(X_training, y_train)
#     predictions = model.predict(X_testing)
    
#     # Convert predictions back to string labels for comparison if needed
#     if needs_numeric:
#         predictions = ["Authentic" if p == 0 else "Counterfeit" for p in predictions]
    
#     correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
#     incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
#     total = len(y_testing)
#     accuracy = 100 * correct / total
    
#     print(f"{model_name}:")
#     print(f"Correct: {correct}")
#     print(f"Incorrect: {incorrect}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print()
    
#     models.append(model_name)
#     results.append({
#         'name': model_name,
#         'correct': correct,
#         'incorrect': incorrect,
#         'accuracy': accuracy,
#         'predictions': predictions,
#         'model': model
#     })

# # Visualization 1: Accuracy Comparison Bar Chart
# plt.figure(figsize=(10, 6))
# accuracies = [r['accuracy'] for r in results]
# colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
# bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
# plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# plt.xlabel('Model', fontsize=12, fontweight='bold')
# plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
# plt.ylim(0, 105)
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on bars
# for bar, acc in zip(bars, accuracies):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#              f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# plt.tight_layout()
# plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
# print("Saved: accuracy_comparison.png")
# plt.close()

# # Visualization 2: Correct vs Incorrect Predictions
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.arange(len(models))
# width = 0.35

# correct_counts = [r['correct'] for r in results]
# incorrect_counts = [r['incorrect'] for r in results]

# bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', 
#                color='#27ae60', edgecolor='black', linewidth=1.2)
# bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
#                color='#e74c3c', edgecolor='black', linewidth=1.2)

# ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
# ax.set_xlabel('Model', fontsize=12, fontweight='bold')
# ax.set_title('Correct vs Incorrect Predictions by Model', fontsize=14, fontweight='bold', pad=20)
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45, ha='right')
# ax.legend(fontsize=11)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 2,
#                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.savefig('correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
# print("Saved: correct_vs_incorrect.png")
# plt.close()

# # Visualization 3: Confusion Matrices for all models
# fig, axes = plt.subplots(3, 3, figsize=(16, 14))
# axes = axes.ravel()

# for idx, result in enumerate(results):
#     cm = confusion_matrix(y_testing, result['predictions'], 
#                           labels=["Authentic", "Counterfeit"])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                                    display_labels=["Authentic", "Counterfeit"])
#     disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
#     axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2f}%", 
#                        fontweight='bold', fontsize=11)
#     axes[idx].grid(False)

# # Hide unused subplots
# for idx in range(len(results), len(axes)):
#     axes[idx].axis('off')

# plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=0.995)
# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
# print("Saved: confusion_matrices.png")
# plt.close()

# # Visualization 4: Feature Distribution (first two features)
# plt.figure(figsize=(12, 8))

# X_training_array = np.array(X_training)
# X_testing_array = np.array(X_testing)

# # Plot training data
# authentic_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                    if y_training[i] == "Authentic"]
# counterfeit_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                      if y_training[i] == "Counterfeit"]

# if authentic_train:
#     authentic_train = np.array(authentic_train)
#     plt.scatter(authentic_train[:, 0], authentic_train[:, 1], 
#                c='green', marker='o', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Authentic (Train)')

# if counterfeit_train:
#     counterfeit_train = np.array(counterfeit_train)
#     plt.scatter(counterfeit_train[:, 0], counterfeit_train[:, 1], 
#                c='red', marker='s', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Counterfeit (Train)')

# plt.xlabel('Feature 1 (Variance)', fontsize=12, fontweight='bold')
# plt.ylabel('Feature 2 (Skewness)', fontsize=12, fontweight='bold')
# plt.title('Banknote Feature Distribution (Training Data)', fontsize=14, fontweight='bold', pad=20)
# plt.legend(fontsize=11)
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
# print("Saved: feature_distribution.png")
# plt.close()

# # Visualization 5: Model Performance Summary
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left plot: Accuracy with error indication
# accuracies = [r['accuracy'] for r in results]
# colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

# bars = ax1.barh(models, accuracies, color=colors_gradient, edgecolor='black', linewidth=1.2)
# ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold', pad=20)
# ax1.set_xlim(0, 105)
# ax1.grid(axis='x', alpha=0.3, linestyle='--')

# for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
#     width = bar.get_width()
#     ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
#              f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

# # Right plot: Pie chart of best model performance
# best_result = max(results, key=lambda x: x['accuracy'])
# sizes = [best_result['correct'], best_result['incorrect']]
# labels = ['Correct', 'Incorrect']
# colors_pie = ['#2ecc71', '#e74c3c']
# explode = (0.05, 0)

# ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
#         autopct='%1.1f%%', shadow=True, startangle=90,
#         textprops={'fontsize': 12, 'fontweight': 'bold'})
# ax2.set_title(f'Best Model: {best_result["name"]}\nAccuracy: {best_result["accuracy"]:.2f}%', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
# print("Saved: performance_summary.png")
# plt.close()

# print("\n✓ All visualizations saved successfully!")
# print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")










# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.linear_model import Perceptron
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import seaborn as sns

# # Read data
# with open('banknotes.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     data = []
#     for row in reader:
#         data.append({
#             "evidence": [float(cell) for cell in row[:4]],
#             "label": "Authentic" if row[4] == "0" else "Counterfeit"
#         })

# # Split data
# holdout = int(0.40 * len(data))
# testing = data[:holdout]
# training = data[holdout:]

# X_training = [row["evidence"] for row in training]
# y_training = [row["label"] for row in training]
# X_testing = [row["evidence"] for row in testing]
# y_testing = [row["label"] for row in testing]

# # Store results
# models = []
# results = []

# # Define models to test
# model_configs = [
#     ("Support Vector Machine", svm.SVC()),
#     ("K-Nearest Neighbors (k=2)", KNeighborsClassifier(n_neighbors=2)),
#     ("Perceptron", Perceptron()),
#     ("Naive Bayes", GaussianNB())
# ]

# # Train and evaluate each model
# for model_name, model in model_configs:
#     model.fit(X_training, y_training)
#     predictions = model.predict(X_testing)
    
#     correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
#     incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
#     total = len(y_testing)
#     accuracy = 100 * correct / total
    
#     print(f"{model_name}:")
#     print(f"Correct: {correct}")
#     print(f"Incorrect: {incorrect}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print()
    
#     models.append(model_name)
#     results.append({
#         'name': model_name,
#         'correct': correct,
#         'incorrect': incorrect,
#         'accuracy': accuracy,
#         'predictions': predictions,
#         'model': model
#     })

# # Visualization 1: Accuracy Comparison Bar Chart
# plt.figure(figsize=(10, 6))
# accuracies = [r['accuracy'] for r in results]
# colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
# bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
# plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# plt.xlabel('Model', fontsize=12, fontweight='bold')
# plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
# plt.ylim(0, 105)
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on bars
# for bar, acc in zip(bars, accuracies):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#              f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# plt.tight_layout()
# plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
# print("Saved: accuracy_comparison.png")
# plt.close()

# # Visualization 2: Correct vs Incorrect Predictions
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.arange(len(models))
# width = 0.35

# correct_counts = [r['correct'] for r in results]
# incorrect_counts = [r['incorrect'] for r in results]

# bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', 
#                color='#27ae60', edgecolor='black', linewidth=1.2)
# bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
#                color='#e74c3c', edgecolor='black', linewidth=1.2)

# ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
# ax.set_xlabel('Model', fontsize=12, fontweight='bold')
# ax.set_title('Correct vs Incorrect Predictions by Model', fontsize=14, fontweight='bold', pad=20)
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45, ha='right')
# ax.legend(fontsize=11)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 2,
#                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.savefig('correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
# print("Saved: correct_vs_incorrect.png")
# plt.close()

# # Visualization 3: Confusion Matrices for all models
# fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# axes = axes.ravel()

# for idx, result in enumerate(results):
#     cm = confusion_matrix(y_testing, result['predictions'], 
#                           labels=["Authentic", "Counterfeit"])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                                    display_labels=["Authentic", "Counterfeit"])
#     disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
#     axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']:.2f}%", 
#                        fontweight='bold', fontsize=11)
#     axes[idx].grid(False)

# plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=0.995)
# plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
# print("Saved: confusion_matrices.png")
# plt.close()

# # Visualization 4: Feature Distribution (first two features)
# plt.figure(figsize=(12, 8))

# X_training_array = np.array(X_training)
# X_testing_array = np.array(X_testing)

# # Plot training data
# authentic_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                    if y_training[i] == "Authentic"]
# counterfeit_train = [X_training_array[i] for i in range(len(X_training_array)) 
#                      if y_training[i] == "Counterfeit"]

# if authentic_train:
#     authentic_train = np.array(authentic_train)
#     plt.scatter(authentic_train[:, 0], authentic_train[:, 1], 
#                c='green', marker='o', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Authentic (Train)')

# if counterfeit_train:
#     counterfeit_train = np.array(counterfeit_train)
#     plt.scatter(counterfeit_train[:, 0], counterfeit_train[:, 1], 
#                c='red', marker='s', s=50, alpha=0.6, 
#                edgecolors='black', linewidth=0.5, label='Counterfeit (Train)')

# plt.xlabel('Feature 1 (Variance)', fontsize=12, fontweight='bold')
# plt.ylabel('Feature 2 (Skewness)', fontsize=12, fontweight='bold')
# plt.title('Banknote Feature Distribution (Training Data)', fontsize=14, fontweight='bold', pad=20)
# plt.legend(fontsize=11)
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
# print("Saved: feature_distribution.png")
# plt.close()

# # Visualization 5: Model Performance Summary
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# # Left plot: Accuracy with error indication
# accuracies = [r['accuracy'] for r in results]
# colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

# bars = ax1.barh(models, accuracies, color=colors_gradient, edgecolor='black', linewidth=1.2)
# ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold', pad=20)
# ax1.set_xlim(0, 105)
# ax1.grid(axis='x', alpha=0.3, linestyle='--')

# for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
#     width = bar.get_width()
#     ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
#              f'{acc:.2f}%', ha='left', va='center', fontweight='bold')

# # Right plot: Pie chart of best model performance
# best_result = max(results, key=lambda x: x['accuracy'])
# sizes = [best_result['correct'], best_result['incorrect']]
# labels = ['Correct', 'Incorrect']
# colors_pie = ['#2ecc71', '#e74c3c']
# explode = (0.05, 0)

# ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
#         autopct='%1.1f%%', shadow=True, startangle=90,
#         textprops={'fontsize': 12, 'fontweight': 'bold'})
# ax2.set_title(f'Best Model: {best_result["name"]}\nAccuracy: {best_result["accuracy"]:.2f}%', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
# print("Saved: performance_summary.png")
# plt.close()

# print("\n✓ All visualizations saved successfully!")
# print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")








# import csv
# from sklearn import svm
# from sklearn.linear_model import Perceptron
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier

# with open('banknotes.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
    
#     data = []
#     for row in reader:
#         data.append({
#             "evidence": [float(cell) for cell in row[:4]],
#             "label": "Authentic" if row[4] == "0" else "Counterfeit"
#         })

# holdout = int(0.40 * len(data))
# testing = data[:holdout]
# training = data[holdout:]

# X_training = [row["evidence"] for row in training]
# y_training = [row["label"] for row in training]
# X_testing = [row["evidence"] for row in testing]
# y_testing = [row["label"] for row in testing]

# model = svm.SVC()
# model.fit(X_training, y_training)
# predictions = model.predict(X_testing)
# correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
# incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
# correct = sum(correct)
# incorrect = sum(incorrect)
# total = len(y_testing)
# print(f"Support Vector Machine:")
# print(f"Correct: {correct}")
# print(f"Incorrect: {incorrect}")
# print(f"Accuracy: {100 * correct / total:.2f}%")
# print()

# model = KNeighborsClassifier(n_neighbors=2)
# model.fit(X_training, y_training)
# predictions = model.predict(X_testing)
# correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
# incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
# correct = sum(correct)
# incorrect = sum(incorrect)
# total = len(y_testing)
# print(f"K-Nearest Neighbors (k=2):")
# print(f"Correct: {correct}")
# print(f"Incorrect: {incorrect}")
# print(f"Accuracy: {100 * correct / total:.2f}%")
# print()

# model = Perceptron()
# model.fit(X_training, y_training)
# predictions = model.predict(X_testing)
# correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
# incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
# correct = sum(correct)
# incorrect = sum(incorrect)
# total = len(y_testing)
# print(f"Perceptron:")
# print(f"Correct: {correct}")
# print(f"Incorrect: {incorrect}")
# print(f"Accuracy: {100 * correct / total:.2f}%")
# print()

# model = GaussianNB()
# model.fit(X_training, y_training)
# predictions = model.predict(X_testing)
# correct = (y_testing[i] == predictions[i] for i in range(len(y_testing)))
# incorrect = (y_testing[i] != predictions[i] for i in range(len(y_testing)))
# correct = sum(correct)
# incorrect = sum(incorrect)
# total = len(y_testing)
# print(f"Naive Bayes:")
# print(f"Correct: {correct}")
# print(f"Incorrect: {incorrect}")
# print(f"Accuracy: {100 * correct / total:.2f}%")
