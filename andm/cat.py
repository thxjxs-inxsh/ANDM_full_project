import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="ANDM Notification Processor")

# 1. Load a true Sentence Transformer built for cosine similarity
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()

CENTROIDS_FILE = Path("centroids.pt")

def get_embedding(text: str) -> torch.Tensor:
    """Generates a mathematically normalized 1x384 sentence vector."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Mean pooling based on attention mask
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # CRITICAL: L2 Normalization ensures cosine similarity is perfectly accurate
        return F.normalize(pooled, p=2, dim=1)

# 2. Load Exemplar Matrices
if CENTROIDS_FILE.exists():
    centroids = torch.load(CENTROIDS_FILE, weights_only=True)
else:
    raise FileNotFoundError(f"{CENTROIDS_FILE.name} not found. Please run build.py first.")

class NotificationPayload(BaseModel):
    title: str
    body: str

class FCMMessage(BaseModel):
    token: str = None
    notification: NotificationPayload

class FCMRoot(BaseModel):
    message: FCMMessage

@app.post("/intercept-notification")
def intercept_notification(payload: FCMRoot):
    try:
        start_time = time.time()
        notif = payload.message.notification
        text = f"{notif.title}. {notif.body}"
        
        # 3. Generate Normalized 1x384 Vector
        vector = get_embedding(text)
        vector_list = vector.squeeze(0).tolist()
        
        # 4. Optimized KNN Cosine Similarity using Matrix Multiplication (Dot Product)
        best_score = -1.0
        category = "Ignore"
        
        for cat, matrix in centroids.items():
            sims = torch.matmul(matrix, vector.T).squeeze()
            
            if sims.ndim == 0:
                highest_sim_for_class = sims.item()
            else:
                highest_sim_for_class = sims.max().item()
            
            if highest_sim_for_class > best_score:
                best_score = highest_sim_for_class
                category = cat

        latency_ms = (time.time() - start_time) * 1000

        return {
            "status": "success",
            "classification": category,
            "confidence_score": round(best_score, 4),
            "vector_dimension": f"1x{len(vector_list)}",
            "vector_sample": vector_list[:5],
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)