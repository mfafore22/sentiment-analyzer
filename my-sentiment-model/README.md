# Movie Sentiment Analyzer

A fullstack web application that classifies movie reviews as Positive or Negative using a fine-tuned DistilBERT model with LoRA.

## Project Overview

This project demonstrates an end-to-end machine learning pipeline:

1. Fine-tuned DistilBERT on the IMDB dataset using LoRA (Low-Rank Adaptation)
2. Built a Flask API to serve predictions
3. Created a SvelteKit frontend for user interaction

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | DistilBERT + LoRA |
| Training | Google Colab, Hugging Face Transformers, PEFT |
| Backend | Flask, PyTorch |
| Frontend | SvelteKit, Tailwind CSS |

## Project Structure
```
sentiment-app/
├── server.py                 # Flask API
├── my-sentiment-model/       # LoRA adapter files
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── frontend/                 # SvelteKit app
    └── src/
        └── routes/
            └── +page.svelte
```

## Model Details

- **Base Model:** distilbert-base-uncased
- **Fine-tuning Method:** LoRA (Parameter-Efficient Fine-Tuning)
- **Dataset:** IMDB Movie Reviews (25,000 training samples)
- **Task:** Binary Sentiment Classification (Positive/Negative)
- **Trainable Parameters:** 739,586 (1.09% of total)
- **Training Time:** ~4 minutes on Google Colab

### Training Results

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 0.3236 | 0.3291 |
| 2 | 0.2881 | 0.3239 |
| 3 | 0.2659 | 0.3179 |

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-app.git
cd sentiment-app
```

### 2. Download the model

Download the LoRA adapter files and place them in the `my-sentiment-model/` folder.

### 3. Install dependencies
```bash
pip install flask flask-cors transformers torch peft
```

### 4. Run the backend
```bash
python server.py
```

The API will be available at `http://127.0.0.1:5000`

### 5. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`

## API Reference

### Analyze Sentiment

**Endpoint:** `POST /api/analyze`

**Request:**
```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 97.85
}
```

### Health Check

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok"
}
```

## How It Works
```
User Input --> SvelteKit Frontend --> Flask API --> LoRA Model --> Prediction
     |                                                                |
     <-------------------------- Response ---------------------------
```

1. User enters a movie review in the frontend
2. Frontend sends POST request to Flask API
3. API tokenizes text and runs it through the model
4. Model returns sentiment prediction with confidence score
5. Frontend displays the result

## Limitations

- Trained only on movie reviews; may not generalize to other domains
- Binary classification only (no neutral option)
- Maximum input length of 512 tokens
- English language only

## Future Improvements

- Deploy to Hugging Face Spaces
- Add multi-class sentiment (positive, neutral, negative)
- Support for other languages
- Batch analysis for multiple reviews

## License

MIT

## Author

Your Name

## Acknowledgments

- Hugging Face for Transformers and PEFT libraries
- IMDB dataset
- Google Colab for free GPU access