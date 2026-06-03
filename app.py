import joblib
import pandas as pd
import random

# ১. AI Model Load
ai_model = joblib.load('trained_supplier_ai.pkl')


categories_config = [
    ('Laptop', 40000, 150000, 1, 10),
    ('Skincare', 500, 5000, 10, 100),
    ('Phone', 15000, 130000, 1, 15),
    ('Furniture', 5000, 90000, 1, 5),
    ('Appliances', 10000, 120000, 1, 3),
    ('Fashion', 300, 6000, 20, 200),
    ('Grocery', 50, 2500, 50, 500),
    ('Sports', 1000, 20000, 2, 20),
    ('Auto', 2000, 60000, 1, 10),
    ('Stationery', 10, 500, 100, 1000),
    ('Toys', 200, 12000, 5, 50),
    ('Industrial', 10000, 250000, 1, 5),
    ('Pets', 500, 10000, 5, 30),
    ('Kitchen', 800, 25000, 2, 20),
    ('Solar', 30000, 500000, 1, 3),
    ('Camera', 25000, 300000, 1, 2),
    ('Gardening', 200, 7000, 10, 50),
    ('Music', 3000, 180000, 1, 4)
]

incoming_data = []


for cat, min_p, max_p, min_m, max_m in categories_config:
    names = [f"Global {cat}", f"Elite {cat} Corp", f"Trust {cat} Ltd", f"Prime {cat} Hub", 
             f"Budget {cat} Shop", f"Metro {cat} Supply", f"Expert {cat} Solutions", 
             f"Sabbir {cat} Center", f"NextGen {cat}", f"Classic {cat} Traders"]
    
    for name in names:
        incoming_data.append({
            'name': name,
            'category': cat,
            'price': random.randint(min_p, max_p),
            'quality': round(random.uniform(6.5, 9.9), 1),
            'reliability': random.randint(75, 99),
            'moq': random.randint(min_m, max_m),
            'delivery_time': random.randint(1, 10),
            'reviews': round(random.uniform(3.5, 5.0), 1)
        })

df_suppliers = pd.DataFrame(incoming_data)


def analyze_category(target_category):
    df_filtered = df_suppliers[df_suppliers['category'] == target_category].copy()

    if df_filtered.empty:
        print(f"❌ No suppliers found for category: {target_category}")
        return

    features = df_filtered[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
    df_filtered['ai_score'] = ai_model.predict(features)

    df_ranked = df_filtered.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
    best_supplier = df_ranked.iloc[0]

    print(f"\n === AI Supplier Ranking (Category: {target_category}) === ")

    for index, row in df_ranked.iterrows():
        rank = index + 1
        final_score = int(row['ai_score'] + 0.5)
        
        if rank <= 3:
            print(f"🌟 Rank #{rank}: {row['name']} (Highly Recommended)")
            print(f"   AI Score: {final_score}%")
            print(f"   Details: Price {row['price']}tk | Quality {row['quality']} | Delivery {row['delivery_time']} days | MOQ {row['moq']}")
            
        else:
            print(f"⚠️ Rank #{rank}: {row['name']} (Not Recommended)")
            print(f"   AI Score: {final_score}%")
        
            reasons = []
            if row['price'] > best_supplier['price']:
                reasons.append(f"Higher Price (+{row['price'] - best_supplier['price']} tk)")
            if row['quality'] < best_supplier['quality']:
                reasons.append("Lower Quality")
            if row['delivery_time'] > best_supplier['delivery_time']:
                reasons.append(f"Slower Delivery (+{row['delivery_time'] - best_supplier['delivery_time']} days)")
            if row['moq'] > best_supplier['moq']:
                reasons.append("Higher Minimum Order (MOQ)")
                
            reason_text = ", ".join(reasons) if reasons else "Overall score is just lower"
            print(f"   Reason for Rejection: {reason_text}")
            
        print("-" * 75)

    print(f"\n💡 AI Conclusion: '{best_supplier['name']}' is the Best Supplier because they offer the optimal balance of all criteria!\n")


analyze_category('Phone')
