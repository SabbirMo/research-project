import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# AI Models Import
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

print("Loading Data...")
df = pd.read_csv('supplier_train_data.csv')

X = df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
y = df['historical_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN এর জন্য ডেটা স্কেলিং
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1), # verbose=-1 to hide extra info
    "ANN (Neural Network)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

print("\nTraining Models... Please wait.\n")
print("-" * 30)

best_accuracy = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    if name == "ANN (Neural Network)":
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
    
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    
    # ANN বাদে বেস্ট মডেল সিলেক্ট করা (ওয়েব অ্যাপের সুবিধার্থে)
    if accuracy > best_accuracy and name != "ANN (Neural Network)":
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

print("-" * 30)
print(f"\nWinner: {best_model_name} with {best_accuracy * 100:.2f}% Accuracy!")

joblib.dump(best_model, 'trained_supplier_ai.pkl')
print("\nSaved the best model as 'trained_supplier_ai.pkl'")
print("Your Web App is now ready to use this advanced model!")