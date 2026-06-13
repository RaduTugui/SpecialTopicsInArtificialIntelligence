# Memotion Multimodal Sarcasm Detector

A Flask web application that detects sarcasm in memes by combining visual features (CLIP) and text features (RoBERTa). Built on the **Memotion 7k** dataset as part of a research project on computational humor, sarcasm and irony detection with a final accuracy of 78%.

---

## What This App Does

Upload a meme image, type its caption, and the model tells you whether the content is **sarcastic** or **not sarcastic**, along with a confidence score. The model fuses:

- **CLIP ViT-B/32** → 512-dimensional image embedding
- **RoBERTa-base** → 768-dimensional text embedding (CLS token)
- **Classifier head** → concatenated 1280-dim vector → 2 output classes

---

## Research Findings (Memotion 7k Corpus)

The following results come from a correlation analysis between sarcasm and emotion/sentiment labels in the dataset.

### Key Statistics

| Test | Result | Interpretation |
|---|---|---|
| Spearman ρ (sarcasm ↔ sentiment) | **+0.05** | Negligible — sarcasm does not predict negativity |
| Cramér's V (chi-square effect size) | **0.092** | Small practical effect |
| Kruskal-Wallis H | **75.02**, p < 0.001 | Groups differ in sentiment distribution |
| Sarcasm ↔ Offensiveness (ρ) | **+0.42** ⭐ | Moderate — strongest finding |
| Sarcasm ↔ Humour (ρ) | **+0.16** | Weak positive |
| Sarcasm ↔ Motivational (ρ) | **+0.17** | Weak positive, counterintuitive |

### Class Distribution

```
Sarcasm:          Sentiment:
general      50%  positive       45%
not_sarc     22%  neutral        32%
twisted      22%  very_positive  15%
very_twisted  6%  negative        7%
                  very_negative   2%
```

### Main Findings

> **1.** Sarcasm level has a **negligible** relationship with sentiment valence (ρ = +0.05). Sarcastic content does not skew negative — if anything, it reads slightly *more* positive than non-sarcastic content.

> **2.** Sarcasm level has a **moderate** relationship with offensiveness (ρ = +0.42). More intense sarcasm strongly predicts more aggressive/hostile content, even when sentiment stays positive. This is the strongest and most defensible result.

> **3.** Sarcasm weakly co-occurs with both humour and motivational content (ρ ≈ +0.16–0.17), confirming sarcasm in this corpus functions primarily as a **comedic register**.

> **4.** Sarcastic memes score *higher* in sentiment than non-sarcastic ones (Mann-Whitney U, p < 0.001) — likely because annotators rated comedic surface tone rather than ironic intent.

### Why These Tests Were Chosen

- **Spearman ρ** — both variables are ordinal, non-normal; monotonic relationship is appropriate
- **Chi-square + Cramér's V** — tests independence between categorical variables; V corrects for large-N inflation of significance (mandatory at n ≈ 7000)
- **Kruskal-Wallis** — non-parametric k-group comparison; does not assume normality
- **Mann-Whitney U** — non-parametric two-group rank test for binary sarcastic vs. not-sarcastic split

---

## Project Structure

```
your_project/
├── app.py                      ← Flask backend
├── memotion_multimodal.pth     ← trained classifier weights (download separately)
├── requirements.txt
└── templates/
    └── index.html              ← web UI
```

---

## Requirements

- Python 3.10+
- pip
- Internet connection (first run only — downloads CLIP and RoBERTa from HuggingFace, ~1.1 GB total, cached after first run)

---

## Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd your_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

For **CPU-only** (recommended if you have no NVIDIA GPU — installs a much smaller PyTorch build):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask flask-cors transformers Pillow scikit-learn
```

Or install everything from `requirements.txt` (may download large CUDA packages on some systems):

```bash
pip install -r requirements.txt --timeout 300
```

### 4. Download the trained model weights

> ⚠️ **This step is required.** The app will crash on startup without it.

Download `memotion_multimodal.pth` from the link below and place it in the **root of the project folder** (next to `app.py`):

📥 **[Download memotion_multimodal.pth](https://drive.google.com/drive/folders/1HaS5sM9b-tLEd1magm8zRgEir6c58MN2)**

> The same Google Drive folder also contains the **Colab training notebook** (`Memotion_Multimodal_Colab.ipynb`) used to train the model on the Memotion 7k dataset. If you want to retrain or fine-tune the model yourself, open that notebook in Google Colab, mount your Drive, and run it end to end — it will produce a new `memotion_multimodal.pth` that you can drop straight into this project.

After downloading, your folder should look like:

```
your_project/
├── app.py
├── memotion_multimodal.pth     ← here
├── requirements.txt
└── templates/
    └── index.html
```

### 5. Run the app

```bash
python app.py
```

On **first run**, CLIP and RoBERTa model weights (~1.1 GB total) will be downloaded automatically from HuggingFace and cached. This takes a few minutes depending on your connection. Every subsequent run starts in seconds.

You will see:

```
[INFO] Using device: cpu
[INFO] Loading CLIP...
[INFO] Loading RoBERTa...
[INFO] Loading classifier weights from memotion_multimodal.pth...
[INFO] Model ready.
 * Running on http://127.0.0.1:5000
```

### 6. Open the UI

Go to **http://localhost:5000** in your browser.

---

## Usage

1. Upload a meme image (JPG, PNG, GIF, WEBP)
2. Type or paste the meme's caption text
3. Click **Analyse sarcasm**
4. The UI shows the predicted label, confidence scores, and feature contribution breakdown

You can also test **text only** (no image) using the example buttons in the UI — the model falls back to a zero image vector in this case.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `EOFError` on startup | The `.pth` file is empty (0 bytes) — re-download it from the Drive link |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your virtual environment |
| Slow first startup | Normal — HuggingFace is downloading CLIP + RoBERTa (~1.1 GB). Cached after first run |
| Port 5000 in use | Change `app.run(port=5000)` to another port (e.g. `5001`) in `app.py` |
| RoBERTa `UNEXPECTED`/`MISSING` key warnings | Safe to ignore — these refer to the LM head, not the encoder used here |

---

## Dataset

**Memotion Dataset 7k** — 6,992 memes with multi-label annotations:

| Column | Values |
|---|---|
| `sarcasm` | not_sarcastic, general, twisted_meaning, very_twisted |
| `overall_sentiment` | very_negative → very_positive |
| `humour` | not_funny, funny, very_funny, hilarious |
| `offensive` | not_offensive, slight, very_offensive, hateful_offensive |
| `motivational` | not_motivational, motivational |

---

## Model Architecture

```
Image → CLIP ViT-B/32 → 512-dim embedding  ─┐
                                              ├─ concat → 1280-dim → Linear(512) → ReLU
Text  → RoBERTa-base → 768-dim CLS token   ─┘                    → Dropout(0.3)
                                                                   → Linear(128) → ReLU
                                                                   → Linear(2)
                                                                   → Softmax → {sarcastic, not_sarcastic}
```

Both CLIP and RoBERTa encoders are **frozen** during training. Only the classifier head was trained on Memotion 7k (80/20 stratified split, 5 epochs, Adam lr=1e-4, CrossEntropyLoss).

---

## Limitations

- The model was trained on internet memes only — performance on other social media formats (tweets, Reddit posts) is untested
- Author-level personality/profile correlation cannot be addressed with Memotion 7k, which has no user identity. For this, a corpus with repeated authors (e.g. SemEval 2018 Task 3, SARC Reddit corpus) would be needed
- Sentiment labels in the dataset likely capture surface tone rather than ironic intent, which may cause the model to underestimate pessimism in deeply sarcastic content

---

## References

- Van Hee et al. (2018) — *SemEval-2018 Task 3: Irony Detection in English Tweets*
- Danescu-Niculescu-Mizil et al. — *A Multidimensional Approach for Detecting Irony in Twitter*
- Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Liu et al. (2019) — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Memotion Dataset — SemEval-2020 Task 8
