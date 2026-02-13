"""
db.py — MongoDB connection and helper functions.

Centralises all database operations so the rest of the app stays clean.
"""

import os
from datetime import datetime, timezone
from urllib.parse import quote_plus
from pymongo import MongoClient

# ── Connection ───────────────────────────────────────────────────────
_USER = "athrv"
_PASS = "athrv1234567"
_HOST = "cluster0.lqhwbhe.mongodb.net"

_DEFAULT_URI = f"mongodb+srv://{quote_plus(_USER)}:{quote_plus(_PASS)}@{_HOST}/?appName=Cluster0"
MONGO_URI = os.environ.get("MONGO_URI", _DEFAULT_URI)

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["plantguard"]          # database name

# ── Collections ──────────────────────────────────────────────────────
predictions_col = db["predictions"]
contacts_col    = db["contacts"]
users_col       = db["users"]

# Ensure unique email index for users
try:
    users_col.create_index("email", unique=True)
except Exception:
    pass  # Index already exists


# ── Helper Functions ─────────────────────────────────────────────────

def save_prediction(image_filename: str, disease_name: str, confidence: float, details: dict) -> str:
    """Save a prediction result to MongoDB. Returns the inserted document ID."""
    doc = {
        "image_filename": image_filename,
        "disease_name":   disease_name,
        "confidence":     confidence,
        "description":    details.get("description", ""),
        "possible_steps": details.get("possible_steps", ""),
        "supplement":     details.get("supplement", {}),
        "created_at":     datetime.now(timezone.utc),
    }
    result = predictions_col.insert_one(doc)
    return str(result.inserted_id)


def get_recent_predictions(limit: int = 20) -> list:
    """Return the most recent predictions, newest first."""
    cursor = predictions_col.find().sort("created_at", -1).limit(limit)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def save_contact_message(name: str, email: str, message: str) -> str:
    """Save a contact-us form submission to MongoDB."""
    doc = {
        "name":       name,
        "email":      email,
        "message":    message,
        "created_at": datetime.now(timezone.utc),
    }
    result = contacts_col.insert_one(doc)
    return str(result.inserted_id)


# ── User Functions ───────────────────────────────────────────────────

def find_user_by_email(email: str):
    """Find a user by email. Returns the document or None."""
    return users_col.find_one({"email": email})


def create_user(name: str, email: str, hashed_password: str) -> str:
    """Create a new user. Returns the inserted document ID."""
    doc = {
        "name":     name,
        "email":    email,
        "password": hashed_password,
        "created_at": datetime.now(timezone.utc),
    }
    result = users_col.insert_one(doc)
    return str(result.inserted_id)


def ping() -> bool:
    """Test the MongoDB connection. Returns True if connected."""
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False
