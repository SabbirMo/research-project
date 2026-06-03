from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

model_name = 'trained_supplier_ai.pkl'
model_path = None
possible_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name),
    os.path.join(os.getcwd(), model_name),
    os.path.join(os.getcwd(), 'api', model_name)
]

for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

try:
    if model_path:
        model = joblib.load(model_path)
        print(f"✅ AI Model Loaded Successfully from: {model_path}")
    else:
        model = joblib.load(model_name)
        print("✅ AI Model Loaded Successfully via default path!")
except Exception as e:
    print(f"❌ Model load error: {e}")

# 🛠️ একাধিক ফাইল সেভ এবং ডিলিট করার জন্য ডিকশনারি তৈরি করা হলো
uploaded_files_data = {}

categories_config = {
    'Laptop': (40000, 150000, 1, 10), 'Phone': (15000, 130000, 1, 20),
    'Skincare': (300, 5000, 20, 200), 'Furniture': (5000, 80000, 1, 5),
    'Appliances': (10000, 100000, 1, 3), 'Fashion': (200, 5000, 50, 500),
    'Grocery': (50, 2000, 100, 1000), 'Sports': (500, 15000, 5, 50),
    'Auto': (1000, 50000, 1, 20), 'Stationery': (5, 500, 500, 5000),
    'Toys': (100, 10000, 10, 100), 'Industrial': (5000, 200000, 1, 5),
    'Pets': (200, 8000, 10, 50), 'Kitchen': (500, 20000, 5, 30),
    'Solar': (20000, 500000, 1, 5), 'Camera': (20000, 250000, 1, 3),
    'Gardening': (100, 5000, 10, 100), 'Music': (2000, 150000, 1, 5)
}

# 🛠️ ফাইল আপলোড API
@app.route('/api/upload_csv', methods=['POST', 'OPTIONS'])
def upload_csv():
    global uploaded_files_data
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part found'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if 'supplier_name' in df.columns:
                df = df.rename(columns={'supplier_name': 'name'})
                
            required_features = ['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']
            if 'category' not in df.columns or 'name' not in df.columns or not all(col in df.columns for col in required_features):
                return jsonify({'error': 'CSV must contain required headers.'}), 400
                
            uploaded_files_data[file.filename] = df.copy()
            
            all_uploaded_df = pd.concat(uploaded_files_data.values(), ignore_index=True)
            unique_categories = all_uploaded_df['category'].dropna().unique().tolist()
            
            return jsonify({
                'message': '✅ Dataset uploaded successfully!',
                'filename': file.filename,
                'new_categories': unique_categories,
                'total_records': len(df)
            })
        return jsonify({'error': 'Invalid file format. Only .csv allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🛠️ ফাইল ডিলিট API
@app.route('/api/delete_csv', methods=['POST', 'OPTIONS'])
def delete_csv():
    global uploaded_files_data
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.json
        filename = data.get('filename')
        
        if filename in uploaded_files_data:
            del uploaded_files_data[filename] # মেমরি থেকে ডিলিট করা হলো
            
            # ডিলিট করার পর বাকি ফাইলগুলো থেকে ক্যাটাগরি আপডেট করা
            if len(uploaded_files_data) > 0:
                all_uploaded_df = pd.concat(uploaded_files_data.values(), ignore_index=True)
                unique_categories = all_uploaded_df['category'].dropna().unique().tolist()
            else:
                unique_categories = []
                
            return jsonify({'message': f'{filename} deleted.', 'new_categories': unique_categories})
            
        return jsonify({'error': 'File not found on server'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# মেইন সার্চ API
@app.route('/api/search_suppliers', methods=['POST', 'OPTIONS'])
def search_suppliers():
    global uploaded_files_data
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.json
        category = data.get('category', 'Laptop')
        
        custom_df = None
        if len(uploaded_files_data) > 0:
            all_uploaded_df = pd.concat(uploaded_files_data.values(), ignore_index=True)
            if category in all_uploaded_df['category'].values:
                custom_df = all_uploaded_df[all_uploaded_df['category'] == category].copy()
                
        if custom_df is not None:
            features = custom_df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
            predictions = model.predict(features)
            custom_df['ai_score'] = [min(max(round(score, 2), 0), 100) for score in predictions]
            df = custom_df.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
        else:
            if category not in categories_config:
                category = 'Laptop'
            min_p, max_p, min_m, max_m = categories_config[category]
            random.seed(category)
            
            suppliers = []
            for i in range(1, 31): 
                suppliers.append({
                    'name': f"Supplier_{random.randint(100, 999)}",
                    'price': random.randint(min_p, max_p),
                    'moq': random.randint(min_m, max_m),
                    'quality': round(random.uniform(6.0, 10.0), 1),
                    'reliability': random.randint(70, 100),
                    'delivery_time': random.randint(1, 14),
                    'reviews': round(random.uniform(3.0, 5.0), 1)
                })
            random.seed()
            df = pd.DataFrame(suppliers)
            features = df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
            predictions = model.predict(features)
            df['ai_score'] = [min(max(round(score, 2), 0), 100) for score in predictions]
            df = df.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
        
        result_list = []
        for index, row in df.iterrows():
            rank = index + 1
            supplier_data = row.to_dict()
            supplier_data['rank'] = rank
            
            if rank <= 3:
                supplier_data['reason'] = 'Top AI Match (Best Balance)'
            elif rank <= 10:
                supplier_data['reason'] = 'Satisfactory overall score'
            else:
                supplier_data['reason'] = 'Price/Quality ratio is low'
                
            result_list.append(supplier_data)

        return jsonify({'suppliers': result_list, 'category': category})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)