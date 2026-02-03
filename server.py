# ============================================
# PART 1: IMPORTS
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel  # NEW: This loads LoRA adapters
import torch


# ============================================
# PART 2: SETUP
# ============================================

app = Flask(__name__)
CORS(app)


# ============================================
# PART 3: LOAD MODEL (Base + LoRA Adapter)
# ============================================

print("Loading model...")

# Step 1: Load the original base model from Hugging Face
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# Step 2: Load YOUR LoRA adapter on top of it
model = PeftModel.from_pretrained(base_model, "./my-sentiment-model")

# Step 3: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Set to evaluation mode
model.eval()

print("Model loaded!")


# ============================================
# PART 4: API ENDPOINT - Analyze Sentiment
# ============================================

@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = outputs.logits.argmax(dim=-1).item()
    
    return jsonify({
        "sentiment": "Positive" if pred == 1 else "Negative",
        "confidence": round(probs.max().item() * 100, 2)
    })


# ============================================
# PART 5: HEALTH CHECK
# ============================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ============================================
# PART 6: START SERVER
# ============================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)