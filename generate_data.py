import pandas as pd
import numpy as np
import random

# ডেটার পরিমাণ ২০,০০০ করা হয়েছে প্রোজেক্টের ওয়েট বাড়ানোর জন্য
num_records = 20000  
categories = [
    'Laptop', 'Skincare', 'Phone', 'Furniture', 'Appliances', 
    'Fashion', 'Grocery', 'Sports', 'Auto', 'Stationery', 
    'Toys', 'Industrial', 'Pets', 'Kitchen', 'Solar', 
    'Camera', 'Gardening', 'Music'
]
data = []

print(f"Generating high-quality dataset with {num_records} records, please wait...")

for i in range(num_records):
    category = random.choice(categories)

    if category == 'Laptop':
        price, moq = random.randint(40000, 150000), random.randint(1, 10)
    elif category == 'Phone':
        price, moq = random.randint(15000, 120000), random.randint(5, 20)
    elif category == 'Skincare':
        price, moq = random.randint(300, 5000), random.randint(20, 200)
    elif category == 'Furniture':
        price, moq = random.randint(5000, 80000), random.randint(1, 5)
    elif category == 'Appliances':
        price, moq = random.randint(10000, 100000), random.randint(1, 3)
    elif category == 'Fashion':
        price, moq = random.randint(200, 5000), random.randint(50, 500)
    elif category == 'Grocery':
        price, moq = random.randint(50, 2000), random.randint(100, 1000)
    elif category == 'Sports':
        price, moq = random.randint(500, 15000), random.randint(5, 50)
    elif category == 'Auto':
        price, moq = random.randint(1000, 50000), random.randint(1, 20)
    elif category == 'Stationery':
        price, moq = random.randint(5, 500), random.randint(500, 5000)
    elif category == 'Toys':
        price, moq = random.randint(100, 10000), random.randint(10, 100)
    elif category == 'Industrial':
        price, moq = random.randint(5000, 200000), random.randint(1, 5)
    elif category == 'Pets':
        price, moq = random.randint(200, 8000), random.randint(10, 50)
    elif category == 'Kitchen':
        price, moq = random.randint(500, 20000), random.randint(5, 30)
    elif category == 'Solar':
        price, moq = random.randint(20000, 500000), random.randint(1, 5)
    elif category == 'Camera':
        price, moq = random.randint(20000, 250000), random.randint(1, 3)
    elif category == 'Gardening':
        price, moq = random.randint(100, 5000), random.randint(10, 100)
    elif category == 'Music':
        price, moq = random.randint(2000, 150000), random.randint(1, 5)

    quality = round(random.uniform(6.0, 10.0), 1)
    reliability = random.randint(70, 100)
    delivery_time = random.randint(1, 14) 
    reviews = round(random.uniform(3.0, 5.0), 1)

    data.append([f"Supplier_{i+1}", category, price, quality, reliability, moq, delivery_time, reviews])

df = pd.DataFrame(data, columns=['name', 'category', 'price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews'])

# গ্লোবাল নরমালাইজেশন লজিক
df['norm_price'] = df['price'].min() / df['price']
df['norm_moq'] = df['moq'].min() / df['moq']
df['norm_delivery'] = df['delivery_time'].min() / df['delivery_time']

df['norm_quality'] = df['quality'] / df['quality'].max()
df['norm_reliability'] = df['reliability'] / df['reliability'].max()
df['norm_reviews'] = df['reviews'] / df['reviews'].max()

w_price, w_qual, w_rel, w_moq, w_del, w_rev = 0.25, 0.25, 0.20, 0.10, 0.15, 0.05

df['historical_score'] = (
    (df['norm_price'] * w_price) + (df['norm_quality'] * w_qual) +
    (df['norm_reliability'] * w_rel) + (df['norm_moq'] * w_moq) +
    (df['norm_delivery'] * w_del) + (df['norm_reviews'] * w_rev)
) * 100

# রিয়েলিস্টিক রেজাল্টের জন্য 1.5 নয়েজ দেওয়া হয়েছে
df['historical_score'] = df['historical_score'] + np.random.normal(0, 1.5, len(df))
df['historical_score'] = df['historical_score'].clip(0, 100).round(2)

final_df = df[['name', 'category', 'price', 'quality', 'reliability', 'moq', 'delivery_time', 'reviews', 'historical_score']]
final_df.to_csv('supplier_train_data.csv', index=False)

print(f"Dataset successfully created with {len(final_df)} records!")