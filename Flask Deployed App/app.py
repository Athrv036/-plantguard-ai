"""
app.py — Flask routes for the Plant Disease Detection application.

All ML logic lives in utils.py; database ops live in db.py.
This file only defines HTTP endpoints.
"""

import os
from datetime import timedelta
from flask import Flask, redirect, render_template, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

from utils import (
    predict,
    predict_with_confidence,
    get_disease_info,
    allowed_file,
    ensure_upload_dir,
    disease_info,
    supplement_info,
    UPLOAD_DIR,
)
import db

app = Flask(__name__)

# ── Auth Config ─────────────────────────────────────────────────────
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET", "super-secret-plantguard-key-change-in-prod")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# ── Page Routes ──────────────────────────────────────────────────────

@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/login-page")
def login_page():
    return render_template("login.html")


@app.route("/signup-page")
def signup_page():
    return render_template("signup.html")


@app.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        image = request.files["image"]
        filename = image.filename
        file_path = os.path.join(UPLOAD_DIR, filename)
        ensure_upload_dir()
        image.save(file_path)

        pred, confidence = predict_with_confidence(file_path)
        info = get_disease_info(pred)

        # Save to MongoDB
        try:
            db.save_prediction(filename, info["disease_name"], confidence, info)
        except Exception as e:
            print(f"[MongoDB] Could not save prediction: {e}")

        return render_template(
            "submit.html",
            title=info["disease_name"],
            desc=info["description"],
            prevent=info["possible_steps"],
            image_url=info["disease_image_url"],
            pred=pred,
            sname=info["supplement"]["name"],
            simage=info["supplement"]["image_url"],
            buy_link=info["supplement"]["buy_link"],
        )


@app.route("/market", methods=["GET", "POST"])
def market():
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )


# ── REST API ─────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept an image upload and return JSON with disease info + confidence.

    Usage:
        curl -X POST -F "image=@leaf.jpg" http://127.0.0.1:5000/api/predict
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided. Send a file with key 'image'."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not allowed_file(image.filename):
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    try:
        ensure_upload_dir()
        file_path = os.path.join(UPLOAD_DIR, image.filename)
        image.save(file_path)

        pred, confidence = predict_with_confidence(file_path)
        info = get_disease_info(pred)

        # Save to MongoDB
        try:
            db.save_prediction(image.filename, info["disease_name"], confidence, info)
        except Exception as e:
            print(f"[MongoDB] Could not save prediction: {e}")

        return jsonify({
            "success": True,
            "prediction": {**info, "confidence": confidence},
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── History & Health ─────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    """Return recent prediction history from MongoDB."""
    try:
        limit = request.args.get("limit", 20, type=int)
        history = db.get_recent_predictions(limit)
        return jsonify({"success": True, "predictions": history}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Check if MongoDB is connected."""
    connected = db.ping()
    return jsonify({
        "status": "ok" if connected else "error",
        "mongodb": "connected" if connected else "disconnected",
    }), 200 if connected else 503

@app.route("/test-db")
def test_db():
    """Quick debug route — shows MongoDB connection status, databases, and a test write/read."""
    from datetime import datetime, timezone
    results = {}

    # 1. Ping
    try:
        db.client.admin.command("ping")
        results["1_ping"] = "✅ SUCCESS — MongoDB is reachable"
    except Exception as e:
        results["1_ping"] = f"❌ FAILED — {e}"
        return jsonify(results), 500

    # 2. List databases
    try:
        results["2_databases"] = db.client.list_database_names()
    except Exception as e:
        results["2_databases"] = f"❌ {e}"

    # 3. List collections in plantguard
    try:
        results["3_collections_in_plantguard"] = db.db.list_collection_names()
    except Exception as e:
        results["3_collections_in_plantguard"] = f"❌ {e}"

    # 4. Insert a test document
    try:
        test_doc = {
            "type": "connection_test",
            "message": "Hello from PlantGuard AI!",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        insert_result = db.predictions_col.insert_one(test_doc)
        results["4_test_insert"] = f"✅ Inserted doc with _id: {insert_result.inserted_id}"
    except Exception as e:
        results["4_test_insert"] = f"❌ {e}"

    # 5. Read it back
    try:
        doc = db.predictions_col.find_one({"type": "connection_test"}, sort=[("_id", -1)])
        if doc:
            doc["_id"] = str(doc["_id"])
        results["5_test_read"] = doc
    except Exception as e:
        results["5_test_read"] = f"❌ {e}"

    # 6. Count documents
    try:
        results["6_total_predictions"] = db.predictions_col.count_documents({})
    except Exception as e:
        results["6_total_predictions"] = f"❌ {e}"

    return jsonify(results), 200


# ── Authentication ─────────────────────────────────────────────────

@app.route("/signup", methods=["POST"])
def signup():
    """Register a new user.
    Body JSON: {"name": "...", "email": "...", "password": "..."}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON."}), 400

    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"success": False, "error": "name, email, and password are required."}), 400

    # Check duplicate
    if db.find_user_by_email(email):
        return jsonify({"success": False, "error": "Email already registered."}), 409

    # Hash & store
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    user_id = db.create_user(name, email, hashed_pw)

    return jsonify({"success": True, "message": "User created successfully.", "user_id": user_id}), 201


@app.route("/login", methods=["POST"])
def login():
    """Authenticate and return a JWT token.
    Body JSON: {"email": "...", "password": "..."}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON."}), 400

    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"success": False, "error": "email and password are required."}), 400

    user = db.find_user_by_email(email)
    if not user:
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    if not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    # Create JWT with user_id as identity
    access_token = create_access_token(identity=str(user["_id"]))
    return jsonify({
        "success": True,
        "message": "Login successful.",
        "access_token": access_token,
        "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"]},
    }), 200


@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    """A protected route — requires a valid JWT in the Authorization header.
    Header: Authorization: Bearer <token>
    """
    current_user_id = get_jwt_identity()
    return jsonify({"success": True, "user_id": current_user_id}), 200


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
