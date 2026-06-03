from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Vercel er jonno Model er sothik path doriye deya
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'trained_supplier_ai.pkl')

try:
    model = joblib.load(model_path)
    print("✅ AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")

categories_config = {
    'Laptop': (40000, 150000, 1, 10),
    'Phone': (15000, 130000, 1, 20),
    'Skincare': (300, 5000, 20, 200),
    'Furniture': (5000, 80000, 1, 5),
    'Appliances': (10000, 100000, 1, 3),
    'Fashion': (200, 5000, 50, 500),
    'Grocery': (50, 2000, 100, 1000),
    'Sports': (500, 15000, 5, 50),
    'Auto': (1000, 50000, 1, 20),
    'Stationery': (5, 500, 500, 5000),
    'Toys': (100, 10000, 10, 100),
    'Industrial': (5000, 200000, 1, 5),
    'Pets': (200, 8000, 10, 50),
    'Kitchen': (500, 20000, 5, 30),
    'Solar': (20000, 500000, 1, 5),
    'Camera': (20000, 250000, 1, 3),
    'Gardening': (100, 5000, 10, 100),
    'Music': (2000, 150000, 1, 5)
}

# ⚠️ KHEAL KORO: Ekhane Route e ekhon /api/search_suppliers deya ache
@app.route('/api/search_suppliers', methods=['POST', 'OPTIONS'])
def search_suppliers():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        data = request.json
        category = data.get('category', 'Laptop')
        
        if category not in categories_config:
            category = 'Laptop'
            
        min_p, max_p, min_m, max_m = categories_config[category]
        
        suppliers = []
        for i in range(1, 11): 
            price = random.randint(min_p, max_p)
            moq = random.randint(min_m, max_m)
            quality = round(random.uniform(6.0, 10.0), 1)
            reliability = random.randint(70, 100)
            delivery_time = random.randint(1, 14)
            reviews = round(random.uniform(3.0, 5.0), 1)
            
            suppliers.append({
                'name': f"Supplier_{random.randint(100, 999)}",
                'price': price,
                'quality': quality,
                'reliability': reliability,
                'moq': moq,
                'delivery_time': delivery_time,
                'reviews': reviews
            })
            
        df = pd.DataFrame(suppliers)
        features = df[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
        
        predictions = model.predict(features)
        df['ai_score'] = [min(max(round(score, 2), 0), 100) for score in predictions]
        
        df = df.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
        
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
                    reasons.append(f"Higher Price (+{int(row['price'] - best_supplier['price'])} BDT)")
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