import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance # ANN এর গ্রাফের জন্য নতুন লাইব্রেরি

print("Loading Data and Training Models for Graphs... Please wait.")

# 1. ডেটা লোড এবং প্রসেসিং
df = pd.read_csv('supplier_train_data.csv')
X = df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
y = df['historical_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    "ANN (Neural Network)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

accuracies = {}
predictions = {}
feature_importances = {}

for name, model in models.items():
    if name == "ANN (Neural Network)":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        feature_importances[name] = model.feature_importances_
    
    predictions[name] = preds
    accuracies[name] = r2_score(y_test, preds) * 100

print("Generating Graphs...")

# ---------------------------------------------------------
# Graph 1: Accuracy Comparison Bar Chart
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
colors = ['#4CAF50', '#2196F3', '#FFC107', '#E91E63']
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)
plt.title('Model Accuracy Comparison (%)', fontsize=15, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(min(accuracies.values()) - 5, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}%', va='bottom', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('1_accuracy_comparison.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Graph 2: Actual vs Predicted Scatter Plot
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, preds) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, preds, alpha=0.4, color='indigo', s=15)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[idx].set_title(f'{name}: Actual vs Predicted', fontweight='bold', fontsize=13)
    axes[idx].set_xlabel('Actual Supplier Score')
    axes[idx].set_ylabel('Predicted Score by AI')
    axes[idx].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('2_actual_vs_predicted.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Graph 3: Feature Importance (For ALL 4 Models)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # ৪টা মডেলের জন্য 2x2 গ্রিড
axes = axes.flatten()
features = X.columns
all_models = ["Random Forest", "XGBoost", "LightGBM", "ANN (Neural Network)"]

for idx, name in enumerate(all_models):
    if name == "ANN (Neural Network)":
        # ANN এর জন্য Permutation Importance ব্যবহার করা হচ্ছে
        result = permutation_importance(models[name], X_test_scaled, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
    else:
        # বাকি ৩টি মডেলের জন্য নরমাল Feature Importance
        importances = feature_importances[name]
        
    indices = np.argsort(importances)
    
    axes[idx].barh(range(len(indices)), importances[indices], color='teal', align='center')
    axes[idx].set_yticks(range(len(indices)), [features[i].capitalize() for i in indices])
    axes[idx].set_title(f'{name} Feature Importance', fontweight='bold')
    axes[idx].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300)
plt.close()

print("\nSuccess! 3 Graph images have been saved in your folder:")
print("1. 1_accuracy_comparison.png")
print("2. 2_actual_vs_predicted.png")
print("3. 3_feature_importance.png")