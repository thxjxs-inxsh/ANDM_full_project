import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

print("[1/3] Loading sentence-transformers/all-MiniLM-L6-v2 model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()

def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(pooled, p=2, dim=1)

RULES_FILE = Path("rules.json")
with open(RULES_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"[2/3] Generating embedding matrices from {RULES_FILE.name}...")
# We will keep the variable name 'centroids' to avoid breaking other logic, 
# but these are now full 2D Exemplar Matrices.
centroids = {}

for category, sample_texts in dataset.items():
    embeddings_list = [get_embedding(text) for text in sample_texts]
    
    # Stack tensors into shape (N, 384). DO NOT average them.
    stacked_matrix = torch.cat(embeddings_list, dim=0)
    centroids[category] = stacked_matrix

output_path = Path("centroids.pt")
torch.save(centroids, output_path)
print(f"[3/3] Successfully generated and saved robust '{output_path.name}'.")

print("\n" + "="*60)
print("RUNNING VALIDATION TEST AGAINST EXEMPLAR MATRICES")
print("="*60)

test_samples = [
    ("Driver has arrived", "Immediate"),
    ("New login detected on Chrome", "Immediate"),
    ("Card declined: Insufficient funds", "Immediate"),
    ("Receipt for your ride #491", "Ignore"),
    ("Rate your driver and leave a tip", "Ignore"),
    ("Monthly bank statement ready", "Ignore"),
    ("Flight gate changed to B12", "Adaptive"),
    ("John sent you a message on Slack", "Adaptive")
]

for text, expected in test_samples:
    vec = get_embedding(text) # Shape: (1, 384)
    scores = {}
    
    for cat, matrix in centroids.items():
        # matrix shape is (N, 384). vec shape is (1, 384).
        # Compute similarities across all N examples at once.
        sims = F.cosine_similarity(vec, matrix, dim=1)
        scores[cat] = sims.max().item() # Take the highest matching score
        
    best_cat = max(scores, key=scores.get)
    status = "PASS" if best_cat == expected else "FAIL"
    print(f"[{status}] Text: '{text}' -> Class: {best_cat} (Score: {scores[best_cat]:.4f})")