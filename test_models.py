import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import glob

print("=" * 70)
print("BANKNOTE AUTHENTICATION - MODEL DEMO")
print("=" * 70)

# Check if saved models exist
if not os.path.exists('saved_models') or not glob.glob('saved_models/*.joblib'):
    print("\n❌ Error: No saved models found!")
    print("Please run the training script first to generate model files.")
    exit()

# Read test data
print("\nLoading test data...")
with open('banknotes.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

# Split data (same as training)
holdout = int(0.40 * len(data))
testing = data[:holdout]
training = data[holdout:]

X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]

# Load all saved models
print("Loading pretrained models...\n")
models = []
results = []

model_files = sorted(glob.glob('saved_models/*.joblib'))

for model_file in model_files:
    model_name = os.path.basename(model_file).replace('.joblib', '').replace('_', ' ')
    
    try:
        model = joblib.load(model_file)
        print(f"✓ Loaded: {model_name}")
        
        # Make predictions
        predictions = model.predict(X_testing)
        
        # Convert numeric predictions to labels if needed
        if isinstance(predictions[0], (int, np.integer)):
            predictions = ["Authentic" if p == 0 else "Counterfeit" for p in predictions]
        
        # Calculate accuracy
        correct = sum(y_testing[i] == predictions[i] for i in range(len(y_testing)))
        incorrect = sum(y_testing[i] != predictions[i] for i in range(len(y_testing)))
        total = len(y_testing)
        accuracy = 100 * correct / total
        
        models.append(model_name)
        results.append({
            'name': model_name,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'predictions': predictions,
            'model': model
        })
        
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")

print(f"\n✓ Successfully loaded {len(results)} models")
print("\n" + "=" * 70)

# Display results
print("\nMODEL PERFORMANCE RESULTS:")
print("=" * 70)
for result in results:
    print(f"\n{result['name']}:")
    print(f"  Correct: {result['correct']}")
    print(f"  Incorrect: {result['incorrect']}")
    print(f"  Accuracy: {result['accuracy']:.2f}%")

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS...")
print("=" * 70)

# Visualization 1: Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
accuracies = [r['accuracy'] for r in results]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Comparison (Loaded Models)', fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, 105)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('demo_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: demo_accuracy_comparison.png")
plt.show()

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
ax.set_title('Correct vs Incorrect Predictions by Model (Loaded Models)', fontsize=14, fontweight='bold', pad=20)
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
plt.savefig('demo_correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
print("✓ Saved: demo_correct_vs_incorrect.png")
plt.show()

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

plt.suptitle('Confusion Matrices for All Models (Loaded Models)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('demo_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: demo_confusion_matrices.png")
plt.show()

# Visualization 4: Feature Distribution (first two features)
plt.figure(figsize=(12, 8))

X_training_array = np.array([row["evidence"] for row in training])
y_training = [row["label"] for row in training]

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
plt.savefig('demo_feature_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: demo_feature_distribution.png")
plt.show()

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
plt.savefig('demo_performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: demo_performance_summary.png")
plt.show()

print("\n✓ All visualizations generated successfully!")

# Interactive Prediction Section
print("\n" + "=" * 70)
print("INTERACTIVE PREDICTION MODE")
print("=" * 70)
print(f"\nBest performing model: {best_result['name']} with {best_result['accuracy']:.2f}% accuracy")
print("\nNow you can test the models with your own input!")
print("\nBanknote features required:")
print("  1. Variance of Wavelet Transformed image")
print("  2. Skewness of Wavelet Transformed image")
print("  3. Curtosis of Wavelet Transformed image")
print("  4. Entropy of image")

print("\nExample values:")
print("  Authentic:    Variance=3.62, Skewness=8.67, Curtosis=-2.81, Entropy=-0.45")
print("  Counterfeit:  Variance=-1.40, Skewness=3.32, Curtosis=-1.39, Entropy=-1.99")

while True:
    print("\n" + "=" * 70)
    response = input("\nWould you like to test a prediction? (y/n): ").strip().lower()
    
    if response != 'y':
        break
    
    try:
        print("\nEnter the banknote features:")
        variance = float(input("  Variance:  "))
        skewness = float(input("  Skewness:  "))
        curtosis = float(input("  Curtosis:  "))
        entropy = float(input("  Entropy:   "))
        
        features = [variance, skewness, curtosis, entropy]
        
        print("\n" + "-" * 70)
        print(f"Testing features: {features}")
        print("-" * 70)
        
        # Get predictions from all models
        print(f"\n{'Model':<30} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 70)
        
        predictions_list = []
        
        for result in results:
            model = result['model']
            
            try:
                prediction = model.predict([features])[0]
                
                # Convert numeric to label
                if isinstance(prediction, (int, np.integer)):
                    prediction = "Authentic" if prediction == 0 else "Counterfeit"
                
                predictions_list.append(prediction)
                
                # Get probability if available
                confidence = "N/A"
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([features])[0]
                    confidence = f"{max(proba)*100:.2f}%"
                
                print(f"{result['name']:<30} {prediction:<15} {confidence:<15}")
                
            except Exception as e:
                print(f"{result['name']:<30} Error: {str(e)}")
        
        # Show consensus
        print("\n" + "=" * 70)
        authentic_count = predictions_list.count("Authentic")
        counterfeit_count = predictions_list.count("Counterfeit")
        
        print(f"CONSENSUS: {authentic_count} models predict Authentic, {counterfeit_count} predict Counterfeit")
        
        if authentic_count > counterfeit_count:
            print(f"🟢 FINAL PREDICTION: Authentic (by majority vote)")
        elif counterfeit_count > authentic_count:
            print(f"🔴 FINAL PREDICTION: Counterfeit (by majority vote)")
        else:
            print(f"⚪ FINAL PREDICTION: Tie - No clear consensus")
        
        print("=" * 70)
        
    except ValueError:
        print("\n❌ Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        break

print("\n" + "=" * 70)
print("Demo completed! Thank you for using the banknote authentication system.")
print("=" * 70)