# Quick Start Guide

## Two Pipelines Available

### 1. **Landmark-Based Pipeline** (Lightweight, Fast Setup)
Uses MediaPipe Face Mesh landmarks directly with GRU

### 2. **ArcFace Pipeline** (Production-Ready, More Robust)
Uses ArcFace embeddings for better generalization

---

## Pipeline 1: Landmark-Based (Quick Start)

### Installation
```bash
pip install -r requirements.txt
```

### Collect Data
```bash
python src/expression_recognition/data/annotations.py \
  --classes neutral,smile,frown \
  --session my_session \
  --landmarks \
  --auto-pack \
  --T 3 \
  --stride 1 \
  --val-split 0.2 \
  --draw
```

**Controls:**
- Press `1`, `2`, `3` to select expression class
- Hold `SPACE` to capture frames
- Press `ESC` when done

### Train Model
```bash
python src/expression_recognition/training/trainer.py \
  --train data/features/train_T3.npz \
  --val data/features/val_T3.npz \
  --epochs 20 \
  --batch-size 8 \
  --outdir models
```

### Run Inference
```bash
python src/expression_recognition/inference/live_gru.py \
  --ckpt models/checkpoints/best.pt \
  --T 3 \
  --features xy \
  --draw \
  --cam 0
```

---

## Pipeline 2: ArcFace (Advanced)

### Additional Installation
```bash
pip install insightface pyarrow
```

### Step-by-Step

#### 1. Extract Face Crops from Video
```bash
python src/expression_recognition/data/video_to_crops.py \
  --video my_recording.mp4 \
  --session day1 \
  --out data/crops \
  --size 112
```

#### 2. Label Your Data
Edit `data/crops/day1/manifest.csv` and add labels in the `label` column:
```csv
frame_idx,crop_path,timestamp,session,label,has_face
0,crops/000000.jpg,0.000,day1,neutral,1
1,crops/000001.jpg,0.033,day1,neutral,1
2,crops/000002.jpg,0.067,day1,smile,1
```

#### 3. Build Embeddings
```bash
python src/expression_recognition/data/build_embeddings.py \
  --crops data/crops \
  --manifest data/crops/day1/manifest.csv \
  --backbone iresnet50 \
  --out embeddings/frame_table.parquet \
  --device cuda
```

#### 4. Create Sequences
```bash
python src/expression_recognition/data/make_sequences.py \
  --frames embeddings/frame_table.parquet \
  --stats embeddings/feature_stats.npz \
  --labels embeddings/labels.json \
  --use-split-column \
  --T 12 \
  --stride 4 \
  --out-train embeddings/train_T12.npz \
  --out-val embeddings/val_T12.npz
```

#### 5. Train GRU Head
```bash
python src/expression_recognition/training/train_gru_embed.py \
  --train embeddings/train_T12.npz \
  --val embeddings/val_T12.npz \
  --hidden 128 \
  --epochs 50 \
  --batch-size 32 \
  --outdir models/gru_embed \
  --export-onnx
```

---

## Current Issue Fix

Your training had an error because the model wasn't learning all classes. This has been fixed in `trainer.py`.

### Re-run Training
```bash
python src/expression_recognition/training/trainer.py \
  --train data/features/train_T3.npz \
  --val data/features/val_T3.npz \
  --epochs 20 \
  --batch-size 8 \
  --outdir models
```

The model should now train successfully!

---

## Which Pipeline Should You Use?

### Use **Landmark-Based** if:
- ✅ Quick prototyping
- ✅ Limited GPU resources
- ✅ Real-time performance critical
- ✅ Small dataset (<1000 samples)

### Use **ArcFace** if:
- ✅ Production deployment
- ✅ Need robust features
- ✅ Have larger dataset (>1000 samples)
- ✅ Want better generalization
- ✅ Have GPU available

---

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "No module named 'insightface'"
```bash
pip install insightface
```

### Camera not working
Try different camera indices:
```bash
--cam 0  # Default
--cam 1  # External webcam
--cam 2  # Another camera
```

### Low training accuracy
- Collect more data (aim for 100+ samples per class)
- Use longer sequences: `--T 5` instead of `--T 3`
- Add more classes or balance your dataset
- Try the ArcFace pipeline for better features

---

## Next Steps

1. **Collect more data** - Aim for 200+ samples per expression
2. **Try data augmentation** - Implement in transforms.py
3. **Deploy your model** - Export to ONNX/TensorRT
4. **Build a web app** - Use FastAPI + Gradio
5. **Fine-tune parameters** - Experiment with model architecture

## Support

See full documentation:
- `README.md` - Project overview
- `ARCFACE_PIPELINE.md` - Advanced pipeline details
- `DEVELOPING.md` - Development notes

Happy expression detecting! 🎭

