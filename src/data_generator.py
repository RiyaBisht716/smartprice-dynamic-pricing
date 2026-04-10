import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_data(num_days=1000, output_path="data/final_pricing_dataset.xlsx"):
    np.random.seed(42)
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    # Base parameters
    base_demand = 100
    price_elasticity = 2.5 # Significant drop if price is higher than competitor
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        day_name = days[date.weekday()]
        
        # Product & Market Features
        cost = np.random.uniform(2.0, 5.0)
        competitor_price = cost * np.random.uniform(1.1, 1.6)
        
        # Scenario variables
        is_weekend = 1 if date.weekday() >= 5 else 0
        is_festival = 1 if np.random.random() < 0.05 else 0
        promo = 1 if np.random.random() < 0.1 else 0
        temperature = 15 + 15 * np.random.random() # 15 to 30 deg
        
        # Generate varied prices for the model to learn from (Exploration)
        # Randomly sample prices around the competitor price
        price = competitor_price * np.random.uniform(0.7, 2.0)
        
        # Calculating Realistic Demand (The Law of Demand)
        # Base demand influenced by context
        current_base = base_demand
        if is_weekend: current_base *= 1.3
        if is_festival: current_base *= 1.5
        if promo: current_base *= 1.4
        if temperature > 25: current_base *= 1.1
        
        # Elasticity: Demand decreases as price rises relative to competitor
        price_ratio = price / competitor_price
        
        # Log-linear demand model: D = Base * exp(-elasticity * (ratio - 1))
        # This ensures demand is always positive but drops sharply
        demand = current_base * np.exp(-price_elasticity * (price_ratio - 1))
        
        # Add some gaussian noise to make it realistic
        demand += np.random.normal(0, demand * 0.05)
        demand = max(0, demand)
        
        data.append({
            "date": date,
            "day_of_week": day_name,
            "cost": round(cost, 2),
            "price": round(price, 2),
            "competitor_price": round(competitor_price, 2),
            "temperature": round(temperature, 1),
            "is_weekend": is_weekend,
            "is_festival": is_festival,
            "promo": promo,
            "demand": int(demand)
        })
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as Excel (requirement)
    df.to_excel(output_path, index=False)
    print(f"[OK] Generated {len(df)} rows of high-quality synthetic data at {output_path}")

if __name__ == "__main__":
    generate_realistic_data()
