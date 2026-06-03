import joblib
import pandas as pd
import random
from flask import Flask, request

# ১. Flask app create
app = Flask(__name__)

# ২. AI Model Load
ai_model = joblib.load('trained_supplier_ai.pkl')


def generate_dynamic_data():
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
    
    data = []
    for cat, min_p, max_p, min_m, max_m in categories_config:
      
        names = [f"Global {cat}", f"Elite {cat} Corp", f"Trust {cat} Ltd", f"Prime {cat} Hub", 
                 f"Budget {cat} Shop", f"Metro {cat} Supply", f"Expert {cat} Solutions", 
                 f"Sabbir {cat} Center", f"NextGen {cat}", f"Classic {cat} Traders"]
        
        for name in names:
            data.append({
                'name': name,
                'category': cat,
                'price': random.randint(min_p, max_p),
                'quality': round(random.uniform(6.5, 9.9), 1),
                'reliability': random.randint(75, 99),
                'moq': random.randint(min_m, max_m),
                'delivery_time': random.randint(1, 10),
                'reviews': round(random.uniform(3.5, 5.0), 1)
            })
    return pd.DataFrame(data)

df_suppliers = generate_dynamic_data()
categories_list = sorted(df_suppliers['category'].unique()) 

@app.route('/', methods=['GET', 'POST'])
def index():
    
    options_html = "".join([f'<option value="{c}">{c}</option>' for c in categories_list])

    
    html_header = f"""
    <div style='font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: auto; padding: 20px; background: #ffffff; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);'>
        <h1 style='text-align: center; color: #2c3e50; margin-bottom: 5px;'>🤖 AI Supplier Sourcing Engine</h1>
        <p style='text-align: center; color: #95a5a6; margin-bottom: 25px;'>Select a category from the menu below to analyze best matches.</p>
        
        <form method="POST" style='text-align: center; margin-bottom: 40px; background: #f8f9fa; padding: 20px; border-radius: 10px;'>
            <label for="category" style="font-weight: bold; margin-right: 10px;">Select Category:</label>
            <select name="category" style='padding: 12px; width: 50%; border-radius: 5px; border: 1px solid #ced4da; outline: none; font-size: 16px;'>
                <option value="" disabled selected>Choose a category...</option>
                {options_html}
            </select>
            <button type="submit" style='padding: 12px 30px; background: #2980b9; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 16px; margin-left: 10px;'>Analyze Now</button>
        </form>
    """
    html_footer = "</div>"

    if request.method == 'POST':
        target_category = request.form.get('category')
        if not target_category:
            return html_header + "<p style='color:orange; text-align:center;'>Please select a category first!</p>" + html_footer
            
        df_filtered = df_suppliers[df_suppliers['category'] == target_category].copy()

        # AI Prediction
        features = df_filtered[['price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews']]
        df_filtered['ai_score'] = ai_model.predict(features)
        
        # Ranking
        df_ranked = df_filtered.sort_values(by='ai_score', ascending=False).reset_index(drop=True)
        best_supplier = df_ranked.iloc[0]

        result_html = f"<div style='border-top: 2px solid #eee; padding-top: 20px;'><h2 style='color: #34495e;'>Ranking Results: {target_category}</h2>"
        
        for index, row in df_ranked.iterrows():
            rank = index + 1
            final_score = int(row['ai_score'] + 0.5) # তোমার রাউন্ডিং লজিক
            
            if rank <= 3:
                result_html += f"""
                <div style='background: #f1f9f5; border-left: 6px solid #27ae60; padding: 15px; margin-bottom: 15px; border-radius: 8px;'>
                    <h3 style='margin: 0; color: #27ae60;'>🌟 Rank #{rank}: {row['name']}</h3>
                    <p style='margin: 8px 0; font-size: 15px;'>
                        <b>AI Match Score: {final_score}%</b> | 
                        Price: {row['price']:,}tk | 
                        Quality: {row['quality']} | 
                        Delivery: {row['delivery_time']} days | 
                        MOQ: {row['moq']}
                    </p>
                </div>"""
            else:
                reasons = []
                if row['price'] > best_supplier['price']: reasons.append(f"Higher Price")
                if row['quality'] < best_supplier['quality']: reasons.append("Lower Quality")
                if row['delivery_time'] > best_supplier['delivery_time']: reasons.append("Slower Delivery")
                reason_text = ", ".join(reasons) if reasons else "Lower overall match"
                
                result_html += f"""
                <div style='background: #fff; border-left: 6px solid #e74c3c; padding: 12px; margin-bottom: 10px; border-radius: 8px; border-top: 1px solid #eee; border-right: 1px solid #eee; border-bottom: 1px solid #eee;'>
                    <h4 style='margin: 0; color: #c0392b;'>⚠️ Rank #{rank}: {row['name']}</h4>
                    <p style='margin: 5px 0; font-size: 14px;'><b>Score: {final_score}%</b> | <span style='color: #7f8c8d;'>Gap: {reason_text}</span></p>
                </div>"""

        result_html += f"<div style='background: #34495e; color: white; padding: 15px; border-radius: 8px; margin-top: 30px; text-align: center; font-size: 18px;'><b>Conclusion:</b> <b>{best_supplier['name']}</b> offers the most optimal balance for your needs.</div></div>"
        return html_header + result_html + html_footer

    return html_header + "<div style='text-align: center; padding: 40px;'><img src='https://cdn-icons-png.flaticon.com/512/3067/3067260.png' width='100' style='opacity: 0.2;'><p style='color: #bdc3c7;'>Select a category and click 'Analyze Now' to see the power of AI.</p></div>" + html_footer

if __name__ == '__main__':
    app.run(debug=True)