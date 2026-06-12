import io
import base64
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    RobertaTokenizer, RobertaModel
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# ── Load frozen encoders ───────────────────────────────────────────────────────
print("[INFO] Loading CLIP...")
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

print("[INFO] Loading RoBERTa...")
tokenizer      = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model  = RobertaModel.from_pretrained('roberta-base').to(device)

# Freeze both encoders (same as training)
for p in clip_model.parameters():   p.requires_grad = False
for p in roberta_model.parameters(): p.requires_grad = False
clip_model.eval()
roberta_model.eval()

# ── Classifier head (must match the architecture used in training) ─────────────
class MemotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),   # 512 (CLIP) + 768 (RoBERTa) = 1280
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, text_feat, image_feat):
        fused = torch.cat([text_feat, image_feat], dim=1)  # (B, 1280)
        return self.classifier(fused)

# ── Load saved weights ─────────────────────────────────────────────────────────
MODEL_PATH = 'memotion_multimodal.pth'

print(f"[INFO] Loading classifier weights from {MODEL_PATH}...")
model = MemotionClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("[INFO] Model ready.")

# ── Helper: extract features ───────────────────────────────────────────────────
def get_text_features(text: str) -> torch.Tensor:
    enc = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        out = roberta_model(
            input_ids=enc['input_ids'].to(device),
            attention_mask=enc['attention_mask'].to(device)
        )
    return out.last_hidden_state[:, 0, :]   # CLS token → (1, 768)


def get_image_features(pil_image: Image.Image) -> torch.Tensor:
    inputs = clip_processor(images=pil_image, return_tensors='pt')
    with torch.no_grad():
        out = clip_model.get_image_features(
            pixel_values=inputs['pixel_values'].to(device)
        )
    # get_image_features returns the pooled output directly as a tensor
    if hasattr(out, 'pooler_output'):
        return out.pooler_output          # (1, 512)
    return out                            # already (1, 512)


def get_blank_image_features() -> torch.Tensor:
    """Fallback when no image is provided — zero vector."""
    return torch.zeros(1, 512).to(device)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
      { "text": "meme caption here",
        "image": "<base64-encoded image string>"  ← optional
      }
    Returns JSON:
      { "label": "sarcastic" | "not_sarcastic",
        "sarcastic": 0.87,
        "not_sarcastic": 0.13
      }
    """
    data = request.get_json(force=True)

    text    = data.get('text', '').strip()
    img_b64 = data.get('image')          # base64 string, may be None

    if not text and not img_b64:
        return jsonify({'error': 'Provide at least text or an image.'}), 400

    # Text features
    text_feat = get_text_features(text) if text else torch.zeros(1, 768).to(device)

    # Image features
    if img_b64:
        try:
            img_bytes = base64.b64decode(img_b64)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_feat = get_image_features(pil_image)
        except Exception as e:
            print(f"[WARN] Could not decode image: {e}")
            image_feat = get_blank_image_features()
    else:
        image_feat = get_blank_image_features()

    # Inference
    with torch.no_grad():
        logits = model(text_feat, image_feat)
        probs  = torch.nn.functional.softmax(logits, dim=1)[0]

    not_sarc_prob = round(float(probs[0]), 4)
    sarc_prob     = round(float(probs[1]), 4)
    label         = 'sarcastic' if sarc_prob > not_sarc_prob else 'not_sarcastic'

    return jsonify({
        'label':         label,
        'sarcastic':     sarc_prob,
        'not_sarcastic': not_sarc_prob
    })


@app.route('/health', methods=['GET'])
def health():
    """Quick sanity-check endpoint."""
    return jsonify({'status': 'ok', 'device': str(device)})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)