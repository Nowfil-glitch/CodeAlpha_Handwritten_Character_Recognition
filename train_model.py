import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_character_recognition():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading Handwritten Digits Dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
    
    # Scale pixels to [0, 1]
    X_scaled = X / 16.0
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Multi-Layer Perceptron (Neural Network) Classifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nMLP Neural Network Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Also evaluate Random Forest for benchmark comparison
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest Benchmark Accuracy: {rf_acc * 100:.2f}%")
    
    best_model = mlp if acc >= rf_acc else rf
    
    # Save artifacts
    model_path = os.path.join(base_dir, 'handwritten_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved trained model to {model_path}")
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix — Handwritten Digit Recognition (Acc: {acc*100:.1f}%)')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Save Sample Predictions Plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        ax.set_title(f"Pred: {y_pred[i]} (True: {y_test[i]})", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'sample_predictions.png'), dpi=300)
    plt.close()
    
    print("Visualizations saved successfully.")

if __name__ == '__main__':
    train_character_recognition()
