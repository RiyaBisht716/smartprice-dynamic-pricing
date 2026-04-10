"""
=============================================================================
 Flask Web Application — Dynamic Pricing Dashboard
=============================================================================
 A beautiful, retailer-friendly web interface for the pricing engine.
 
 Endpoints:
   GET  /           → Main dashboard
   POST /api/predict → JSON API for price recommendation
   GET  /api/health  → Health check
=============================================================================
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_compress import Compress

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pricing_engine import PricingEngine

app = Flask(__name__)
# Enable GZip compression for all eligible responses
Compress(app)

# ─── Load pricing engine once at startup ──────────────────────────────────────
try:
    engine = PricingEngine()
    MODEL_LOADED = True
    print(f"  [OK] Pricing engine loaded - model: {engine.model_name}")
except Exception as e:
    MODEL_LOADED = False
    engine = None
    print(f"  [ERR] Failed to load model: {e}")
    print("  -> Run 'python src/model_training.py' first to train the model.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Performance & Caching Optimization
# ═══════════════════════════════════════════════════════════════════════════════

@app.after_request
def add_header(response):
    """Set cache-control headers to reduce load time and optimize static files/pages."""
    if request.path.startswith('/api/'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        # Cache static files and templates for 1 hour
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response

# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html", model_loaded=MODEL_LOADED, current_page="dashboard")

@app.route("/elasticity")
def elasticity():
    """Serve the Elasticity Analysis page."""
    return render_template("elasticity.html", model_loaded=MODEL_LOADED, current_page="elasticity")

@app.route("/ab-testing")
def ab_testing():
    """Serve the A/B Testing experimentation page."""
    return render_template("ab_testing.html", model_loaded=MODEL_LOADED, current_page="ab_testing")

@app.route("/insights")
def insights():
    """Serve the Feature Insights page."""
    return render_template("insights.html", model_loaded=MODEL_LOADED, current_page="insights")

@app.route("/trends")
def trends():
    """Serve the Trends / Time Series page."""
    return render_template("trends.html", model_loaded=MODEL_LOADED, current_page="trends")

@app.route("/alerts")
def alerts():
    """Serve the Price Alerts page."""
    return render_template("alerts.html", model_loaded=MODEL_LOADED, current_page="alerts")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept product features → return optimal price recommendation.
    
    Expected JSON body:
    
        "cost": 2.50,
        "temperature": 25,
        "is_weekend": 0,
        "is_festival": 0,
        "promo": 1,
        "competitor_price": 3.50,
        "day_of_week": "Wednesday"
    }
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    try:
        data = request.get_json()

        # 🔥 DEBUG INPUT
        print("INPUT:", data)

        result = engine.recommend_price(
            cost=float(data["cost"]),
            temperature=float(data["temperature"]),
            is_weekend=int(data["is_weekend"]),
            is_festival=int(data["is_festival"]),
            promo=int(data["promo"]),
            competitor_price=float(data["competitor_price"]),
            day_of_week=str(data["day_of_week"]),
        )

        # 🔥 DEBUG OUTPUT
        print("RESULT:", result)

        return jsonify({"success": True, **result})

    except KeyError as e:
        print("ERROR (KeyError):", e)
        return jsonify({"error": f"Missing field: {e}"}), 400

    except Exception as e:
        print("ERROR (Exception):", e)   # 🔥 VERY IMPORTANT
        return jsonify({"error": str(e)}), 500


@app.route("/api/ab-test", methods=["POST"])
def ab_test():
    """
    Simulate A/B test between two prices.
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    try:
        data = request.get_json()
        result = engine.ab_test_prices(
            price_a=float(data["price_a"]),
            price_b=float(data["price_b"]),
            cost=float(data["cost"]),
            temperature=float(data["temperature"]),
            is_weekend=int(data["is_weekend"]),
            is_festival=int(data["is_festival"]),
            promo=int(data["promo"]),
            competitor_price=float(data["competitor_price"]),
            day_of_week=str(data["day_of_week"]),
        )
        return jsonify({"success": True, **result})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/insights", methods=["GET"])
def get_insights():
    """
    Return feature importances from the model.
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    try:
        result = engine.get_feature_importances()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_name": engine.model_name if MODEL_LOADED else None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # PRODUCTION PERFORMANCE NOTE:
    # Set debug=False and use threading for better lightweight concurrency.
    # For a real production environment, use a WSGI server like Gunicorn or Waitress.
    # e.g., `pip install waitress` -> `waitress-serve --port=5000 app:app`
    app.run(debug=True, host="0.0.0.0", port=5000)
