from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# এআই মডেল পাথ সেটাপ
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

# গ্লোবাল ভেরিয়েবল যেখানে আপলোড করা CSV এর ডাটা সাময়িকভাবে জমা থাকবে
uploaded_suppliers_df = None

# ১৮টি অরিজিনাল ক্যাটাগরি কনফিগারেশন
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

# 🛠️ ১. নতুন API: CSV ফাইল আপলোড এবং প্রসেস করা
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    global uploaded_suppliers_df
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and file.filename.endswith('.csv'):
            # CSV ফাইলটি রিড করা
            df = pd.read_csv(file)
            
            # প্রয়োজনীয় কলামগুলো চেক করা
            required_cols = ['supplier_name', 'category', 'price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']
            if not all(col in df.columns for col in required_cols):
                return jsonify({'error': f'CSV must contain headers: {", ".join(required_cols)}'}), 400
            
            # মেমরিতে সেভ করা
            uploaded_suppliers_df = df.copy()
            
            # ইউনিক ক্যাটাগরিগুলো বের করা ফ্রন্টএন্ডে পাঠানোর জন্য
            unique_categories = df['category'].dropna().unique().tolist()
            
            return jsonify({
                'message': '✅ CSV Uploaded & Processed successfully!',
                'new_categories': unique_categories,
                'total_records': len(df)
            })
            
        return jsonify({'error': 'Invalid file format. Only .csv allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ২. মেইন সার্চ API (আপডেটেড: এটি এখন আপলোড করা ডাটাও চেক করবে)
@app.route('/api/search_suppliers', methods=['POST', 'OPTIONS'])
def search_suppliers():
    global uploaded_suppliers_df
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        data = request.json
        category = data.get('category', 'Laptop')
        
        # 🛠️ চেক করা হচ্ছে ক্যাটাগরিটি আপলোড করা CSV থেকে এসেছে কি না
        if uploaded_suppliers_df is not None and category in uploaded_suppliers_df['category'].values:
            # আপলোড করা CSV থেকে ডাটা ফিল্টার করা
            cat_df = uploaded_suppliers_df[uploaded_suppliers_df['category'] == category].copy()
            
            # কলামের নাম ম্যাচ করানো মডেল ফিচারের সাথে
            features = cat_df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
            predictions = model.predict(features)
            cat_df['ai_score'] = [min(max(round(score, 2), 0), 100) for score in predictions]
            
            # রিনেম করে ক্লায়েন্ট ফরম্যাটে নেওয়া
            cat_df = cat_df.rename(columns={'supplier_name': 'name'})
            df = cat_df.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
            
        else:
            # যদি কাস্টম না হয়, তবে আমাদের আগের ১৮টি রেডিমেড জেনারেটর লজিক চলবে
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
        
        # রেসপন্স ডাটা মেকিং (আগের মতোই)
        best_supplier = df.iloc[0]
        result_list = []
        for index, row in df.iterrows():
            rank = index + 1
            supplier_data = row.to_dict()
            supplier_data['rank'] = rank
            
            if rank <= 3:
                supplier_data['status'] = 'Recommended'
                supplier_data['reason'] = 'Top AI Match (Best Balance)'
            else:
                supplier_data['status'] = 'Not Recommended'
                reasons = []
                if row['price'] > best_supplier['price']:
                    reasons.append("Higher Price")
                if row['quality'] < best_supplier['quality']:
                    reasons.append("Lower Quality")
                if row['delivery_time'] > best_supplier['delivery_time']:
                    reasons.append("Slower Delivery")
                supplier_data['reason'] = ", ".join(reasons) if reasons else "Overall score is lower"
                
            result_list.append(supplier_data)

        return jsonify({'suppliers': result_list, 'category': category})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)