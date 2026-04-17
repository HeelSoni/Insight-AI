import pandas as pd
import numpy as np
import os

def generate_data(filepath="sample_sales_weather.csv"):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-05-31", freq="D")
    n = len(dates)
    
    # Temperature (seasonal, sine wave + noise)
    day_of_year = dates.dayofyear
    temperature_c = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, n)
    
    # Rainfall (random with spikes)
    rainfall_mm = np.where(np.random.rand(n) > 0.8, np.random.exponential(5, n), 0)
    
    # Base sales (negatively correlated with temp)
    sales_units = 500 - 5 * temperature_c + np.random.normal(0, 50, n)
    
    # Lagged rainfall effect (lag 2) positively correlated
    shifted_rainfall = np.roll(rainfall_mm, 2)
    shifted_rainfall[:2] = 0
    sales_units += 10 * shifted_rainfall
    
    sales_units = np.clip(sales_units, 50, 2000).astype(int)
    
    ad_spend_usd = np.random.uniform(100, 1000, n)
    sales_units += (ad_spend_usd * 0.1).astype(int)
    
    foot_traffic = sales_units * 3 + np.random.normal(0, 100, n)
    foot_traffic = np.clip(foot_traffic, 100, 10000).astype(int)
    
    is_holiday = np.where(np.random.rand(n) > 0.9, 1, 0)
    
    category = np.random.choice(["Electronics", "Clothing", "Food"], n)
    segment = np.random.choice(["New", "Returning", "VIP"], n)
    
    df = pd.DataFrame({
        "date": dates,
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "sales_units": sales_units,
        "ad_spend_usd": ad_spend_usd,
        "foot_traffic": foot_traffic,
        "day_of_week": dates.day_name(),
        "is_holiday": is_holiday,
        "product_category": category,
        "customer_segment": segment
    })
    
    df.loc[df['product_category'] == 'Food', 'sales_units'] += np.where(df.loc[df['product_category'] == 'Food', 'day_of_week'].isin(["Saturday", "Sunday"]), 200, 0)
    df.loc[df['product_category'] == 'Electronics', 'sales_units'] += np.where(df.loc[df['product_category'] == 'Electronics', 'day_of_week'].isin(["Tuesday", "Wednesday"]), 100, 0)
    
    df.loc[df['customer_segment'] == 'VIP', 'sales_units'] += 200
    df.loc[df['customer_segment'] == 'VIP', 'foot_traffic'] -= 50
    
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    generate_data("C:\\Users\\Heel\\OneDrive\\Desktop\\Insight AI\\insight_ai\\data\\sample_sales_weather.csv")
