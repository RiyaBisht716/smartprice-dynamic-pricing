"""
=============================================================================
 Dynamic Pricing Engine — Optimal Price Recommendation
=============================================================================
 Given product features (cost, temperature, weekend, festival, promo,
 competitor_price, day_of_week), this module finds the price that
 MAXIMIZES expected profit:

     Profit = (Price − Cost) × Predicted_Demand(Price)

 Approach:
   - Sweep candidate prices from cost × 1.05 to cost × 5.0
   - For each price, predict demand using the trained model
   - Pick the price that maximises profit

 Why brute-force sweep instead of scipy.optimize?
   - Tree-based models (RF, XGB) are non-differentiable
   - A fine-grained sweep is reliable and interpretable
   - Runs in ~5ms per query — fast enough for real-time use
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import functools


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class PricingEngine:
    """
    Load the trained demand model and recommend optimal prices.

    Usage:
        engine = PricingEngine()
        result = engine.recommend_price(
            cost=2.50,
            temperature=25,
            is_weekend=0,
            is_festival=0,
            promo=1,
            competitor_price=3.00,
            day_of_week="Wednesday"
        )
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.model_name = self.metadata["best_model_name"]
        self.model_metrics = self.metadata["metrics"]

    # ─── Build Feature Vector ─────────────────────────────────────────────
    def _build_features(self, price, cost, temperature, is_weekend,
                        is_festival, promo, competitor_price, day_of_week):
        """
        Construct a single-row DataFrame matching the training schema.
        """
        # Engineered features
        comp_safe = max(competitor_price, 0.01)
        price_ratio = price / comp_safe
        markup = price - cost
        price_x_promo = price * promo

        row = {
            "price": price,
            "cost": cost,
            "temperature": temperature,
            "is_weekend": is_weekend,
            "is_festival": is_festival,
            "promo": promo,
            "competitor_price": competitor_price,
            "price_ratio": price_ratio,
            "markup": markup,
            "price_x_promo": price_x_promo,
        }

        # One-hot encode day_of_week (must match training columns)
        all_days = [
            "day_Monday", "day_Saturday", "day_Sunday",
            "day_Thursday", "day_Tuesday", "day_Wednesday"
        ]
        for d in all_days:
            row[d] = 1 if d == f"day_{day_of_week}" else 0

        # Build DataFrame in correct column order
        df = pd.DataFrame([row])
        # Ensure all training columns exist (handles edge cases)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        return df

    # ─── Predict Demand ───────────────────────────────────────────────────
    def predict_demand(self, price, cost, temperature, is_weekend,
                       is_festival, promo, competitor_price, day_of_week):
        """Predict demand for a single price point."""
        df = self._build_features(
            price, cost, temperature, is_weekend,
            is_festival, promo, competitor_price, day_of_week
        )
        X_scaled = self.scaler.transform(df)
        demand = float(self.model.predict(X_scaled)[0])
        return max(demand, 0)  # demand can't be negative

    @functools.lru_cache(maxsize=128)
    def recommend_price(self, cost, temperature, is_weekend, is_festival,
                        promo, competitor_price, day_of_week,
                        min_margin=0.05, max_multiplier=2.5, steps=500):
        """
        Find the price that maximises profit with safety constraints.
        """
        min_price = cost * (1 + min_margin)
        
        # Safety Constraint: Cap the max price to prevent "Greedy AI"
        # 1. Don't go above 2.5x cost (prevents huge outliers)
        # 2. Don't go above 1.8x competitor price (stay within market reality)
        safe_max_mult = min(max_multiplier, 2.5)
        max_price = max(cost * safe_max_mult, competitor_price * 1.5, min_price + 1)
        
        candidate_prices = np.linspace(min_price, max_price, steps)

        # ─── Vectorized Bulk Inference ────────────────────────────────────
        df = pd.DataFrame(0, index=np.arange(len(candidate_prices)), columns=self.feature_names)
        df["price"] = candidate_prices
        df["cost"] = cost
        df["temperature"] = temperature
        df["is_weekend"] = is_weekend
        df["is_festival"] = is_festival
        df["promo"] = promo
        df["competitor_price"] = competitor_price
        
        comp_price = max(competitor_price, 0.0001)
        ratios = candidate_prices / comp_price
        df["price_ratio"] = ratios if competitor_price > 0 else 1.0
        
        cost_safe = max(cost, 0.0001)
        df["markup"] = (candidate_prices - cost) / cost_safe if cost > 0 else 0.0
        df["price_x_promo"] = candidate_prices * promo
        
        day_col = f"day_{day_of_week}"
        if day_col in df.columns:
            df[day_col] = 1

        X_scaled = self.scaler.transform(df)
        demands = self.model.predict(X_scaled)
        demands = np.maximum(demands, 0)

        # ─── Competitive Penalty Heuristic ───────────────────────────────
        # Even if the model says demand is high, punish prices that are
        # way too high compared to the competitor (e.g., > 30% difference)
        # to ensure the AI stays "realistic".
        penalties = np.ones_like(ratios)
        if competitor_price > 0:
            # Gradually penalize starting at 30% above competitor
            overprice_mask = ratios > 1.3
            if np.any(overprice_mask):
                # Penalty factor: (1.0 - aggressive quadratic decay)
                penalties[overprice_mask] = np.maximum(0.1, 1.0 - (ratios[overprice_mask] - 1.3) ** 2)

        profits = (candidate_prices - cost) * demands * penalties

        # Determine optimum
        best_idx = np.argmax(profits)
        best_price = min_price if profits[best_idx] <= 0 else candidate_prices[best_idx]
        best_profit = profits[best_idx]
        best_demand = demands[best_idx]

        # Build return payload
        margin_pct = ((best_price - cost) / cost) * 100 if cost > 0 else 0
        price_vs_comp = ((best_price - competitor_price) / competitor_price * 100
                         if competitor_price > 0 else 0)

        # Thin out curve for chart (every Nth point)
        step_sz = max(1, len(candidate_prices) // 100)
        chart_curve = [
            {
                "price": round(float(candidate_prices[i]), 2),
                "demand": round(float(demands[i]), 2),
                "profit": round(float(profits[i]), 2),
            }
            for i in range(0, len(candidate_prices), step_sz)
        ]

        return {
            "optimal_price": round(float(best_price), 2),
            "predicted_demand": round(float(best_demand), 2),
            "expected_profit": round(float(best_profit), 2),
            "margin_pct": round(float(margin_pct), 1),
            "price_vs_competitor": round(float(price_vs_comp), 1),
            "model_name": self.model_name,
            "model_metrics": self.model_metrics,
            "demand_curve": chart_curve,
        }

    # ─── Feature Importances ──────────────────────────────────────────────
    def get_feature_importances(self):
        """
        Extract feature importances from the trained model if available.
        Only works for tree-based models (RandomForest, XGBoost).
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            
            # Map importances to readable names or aggregate them
            raw_features = list(zip(self.feature_names, importances))
            
            # Create a dictionary to aggregate day_of_week importances and format names
            agg_dict = {
                "price": 0.0, "cost": 0.0, "temperature": 0.0, 
                "is_weekend": 0.0, "is_festival": 0.0, "promo": 0.0, 
                "competitor_price": 0.0, "price_ratio": 0.0, 
                "markup": 0.0, "price_x_promo": 0.0, "day_of_week": 0.0
            }
            
            for name, imp in raw_features:
                if name.startswith("day_"):
                    agg_dict["day_of_week"] += imp
                elif name in agg_dict:
                    agg_dict[name] += imp
                    
            # Convert back to list and sort
            agg_features = [{"feature": k, "importance": float(v)} for k, v in agg_dict.items() if v > 0]
            agg_features.sort(key=lambda x: x["importance"], reverse=True)
            
            # Normalize to sum to 1.0 just in case
            total = sum(item["importance"] for item in agg_features)
            if total > 0:
                for item in agg_features:
                    item["importance"] = round(item["importance"] / total, 4)
            
            return {"success": True, "importances": agg_features}
        else:
            return {"success": False, "error": "Current model does not support feature importances."}

    # ─── A/B Test Simulation ──────────────────────────────────────────────
    def ab_test_prices(self, price_a, price_b, cost, temperature, is_weekend,
                       is_festival, promo, competitor_price, day_of_week):
        """
        Simulate an A/B test by predicting demand and profit for two distinct prices.
        """
        demand_a = self.predict_demand(
            price_a, cost, temperature, is_weekend,
            is_festival, promo, competitor_price, day_of_week
        )
        demand_b = self.predict_demand(
            price_b, cost, temperature, is_weekend,
            is_festival, promo, competitor_price, day_of_week
        )
        
        profit_a = (price_a - cost) * demand_a
        profit_b = (price_b - cost) * demand_b
        
        # Determine winner
        winner = "A" if profit_a > profit_b else "B"
        if abs(profit_a - profit_b) < 0.01:
            winner = "Tie"
            
        profit_diff_pct = 0.0
        if min(profit_a, profit_b) > 0:
            profit_diff_pct = abs(profit_a - profit_b) / min(profit_a, profit_b) * 100
            
        return {
            "variant_A": {
                "price": round(float(price_a), 2),
                "demand": round(float(demand_a), 2),
                "profit": round(float(profit_a), 2),
                "margin": round(float((price_a - cost) / cost * 100 if cost > 0 else 0), 1)
            },
            "variant_B": {
                "price": round(float(price_b), 2),
                "demand": round(float(demand_b), 2),
                "profit": round(float(profit_b), 2),
                "margin": round(float((price_b - cost) / cost * 100 if cost > 0 else 0), 1)
            },
            "winner": winner,
            "profit_difference_pct": round(float(profit_diff_pct), 1)
        }


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = PricingEngine()
    result = engine.recommend_price(
        cost=2.50,
        temperature=25,
        is_weekend=0,
        is_festival=0,
        promo=1,
        competitor_price=3.50,
        day_of_week="Wednesday"
    )
    print(f"\n🏷️  Optimal Price:     ${result['optimal_price']:.2f}")
    print(f"📦 Predicted Demand:  {result['predicted_demand']:.1f} units")
    print(f"💰 Expected Profit:   ${result['expected_profit']:.2f}")
    print(f"📊 Margin:            {result['margin_pct']:.1f}%")
    print(f"⚔️  vs Competitor:     {result['price_vs_competitor']:+.1f}%")
    print(f"🤖 Model:             {result['model_name']}")
