# ArcFace-Based Expression Recognition Pipeline

This pipeline uses ArcFace embeddings with a temporal GRU head for robust expression recognition.

## Overview

Instead of training on raw facial landmarks, this pipeline:
1. Extracts aligned face crops from video
2. Computes frozen ArcFace embeddings (512D from iResNet-50)
3. Trains a lightweight GRU temporal head on embedding sequences
4. Optionally finetunes the last block of the backbone

## Pipeline Steps

### A1. Extract Aligned Face Crops

Extract and align face crops from video using InsightFace.

```bash
python src/expression_recognition/data/video_to_crops.py \
  --video data/sessions/day1/raw.mp4 \
  --session day1 \
  --classes neutral,smile,frown,surprise \
  --out data/crops \
  --size 112
```

**Output:**
- `data/crops/day1/crops/` - Aligned 112x112 face crops
- `data/crops/day1/manifest.csv` - Frame metadata

### A2. Build Embeddings

Convert crops to ArcFace embeddings using frozen iResNet-50.

```bash
python src/expression_recognition/data/build_embeddings.py \
  --crops data/crops \
  --manifest data/crops/day1/manifest.csv \
  --backbone iresnet50 \
  --out embeddings/frame_table.parquet \
  --stats-out embeddings/feature_stats.npz \
  --labels-out embeddings/labels.json \
  --batch-size 32 \
  --device cuda
```

**Output:**
- `embeddings/frame_table.parquet` - Per-frame embeddings with labels
- `embeddings/feature_stats.npz` - Mean and std for standardization
- `embeddings/labels.json` - Class name to ID mapping

### A3. Make Sequences

Create temporal sequences from embeddings for GRU training.

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

**Or with explicit sessions:**
```bash
python src/expression_recognition/data/make_sequences.py \
  --frames embeddings/frame_table.parquet \
  --stats embeddings/feature_stats.npz \
  --labels embeddings/labels.json \
  --sessions-train day1,day2 \
  --sessions-val day3 \
  --T 12 \
  --stride 4 \
  --out-train embeddings/train_T12.npz \
  --out-val embeddings/val_T12.npz
```

**Output:**
- `embeddings/train_T12.npz` - Training sequences (N, T=12, E=512)
- `embeddings/val_T12.npz` - Validation sequences

### B. Train GRU Temporal Head

Train the GRU classifier on pre-computed embeddings.

#### Basic Training (Frozen Backbone)

```bash
python src/expression_recognition/training/train_gru_embed.py \
  --train embeddings/train_T12.npz \
  --val embeddings/val_T12.npz \
  --hidden 128 \
  --layers 1 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --outdir models/gru_embed \
  --export-onnx
```

#### With Finetuning (Last Block)

```bash
python src/expression_recognition/training/train_gru_embed.py \
  --train embeddings/train_T12.npz \
  --val embeddings/val_T12.npz \
  --hidden 128 \
  --layers 1 \
  --backbone iresnet50 \
  --finetune-last-block \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --outdir models/gru_embed_ft \
  --export-onnx
```

**Output:**
- `models/gru_embed/checkpoints/best.pt` - Best checkpoint
- `models/gru_embed/exported/gru_embed.onnx` - ONNX model
- `models/gru_embed/exported/metadata.json` - Model metadata

## Data Format

### Face Crops
- **Size**: 112x112 pixels (default)
- **Format**: JPEG
- **Alignment**: 5-point (eyes, nose, mouth corners)

### Embeddings
- **Dimension**: 512 (iResNet-50) or 512 (iResNet-100)
- **Normalization**: L2 normalized
- **Standardization**: Z-score using training set statistics

### Sequences
- **Shape**: (N, T, E)
  - N: Number of sequences
  - T: Temporal length (e.g., 12 frames)
  - E: Embedding dimension (512)
- **Labels**: Integer class IDs
- **Format**: NPZ (compressed NumPy)

## Model Architecture

### GRU Temporal Head
```
Input: (B, T, 512) embeddings
  ↓
GRU(512 → 128, 1 layer)
  ↓
LayerNorm + Dropout
  ↓
Linear(128 → num_classes)
  ↓
Output: (B, num_classes) logits
```

### With Finetuning
```
Input: (B, T, 3, 112, 112) images
  ↓
ArcFace iResNet-50 (last block trainable)
  ↓
(B, T, 512) embeddings
  ↓
GRU Temporal Head
  ↓
Output: (B, num_classes) logits
```

## Advantages

1. **Robust Features**: ArcFace embeddings are pre-trained on millions of faces
2. **Lightweight**: Only train small GRU head (~100K params)
3. **Fast Training**: Pre-computed embeddings = no backbone forward passes
4. **Optional Finetuning**: Adapt last block to your specific expressions
5. **Easy Export**: ONNX and TensorRT support

## Performance Tips

### For Small Datasets (<1000 samples)
- Use frozen backbone (no finetuning)
- Smaller GRU: `--hidden 64 --layers 1`
- Higher dropout: `--dropout 0.2`
- More data augmentation

### For Large Datasets (>10K samples)
- Finetune last block: `--finetune-last-block`
- Larger GRU: `--hidden 256 --layers 2 --bidirectional 1`
- Lower dropout: `--dropout 0.1`
- Longer sequences: `--T 24`

### For Real-time Inference
- Shorter sequences: `--T 6` or `--T 8`
- Larger stride during training: `--stride 6`
- Export to TensorRT FP16
- Use smaller backbone: iResNet-50 instead of iResNet-100

## TensorRT Export (Optional)

After ONNX export, convert to TensorRT for maximum speed:

```bash
# Install TensorRT
pip install tensorrt

# Convert ONNX to TensorRT
python -c "
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX
with open('models/gru_embed/exported/gru_embed.onnx', 'rb') as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
engine = builder.build_serialized_network(network, config)

# Save engine
with open('models/gru_embed/exported/gru_embed.trt', 'wb') as f:
    f.write(engine)
"
```

## Complete Example Workflow

```bash
# 1. Extract crops from video
python src/expression_recognition/data/video_to_crops.py \
  --video my_recording.mp4 \
  --session day1 \
  --out data/crops

# 2. Label the frames (manually edit manifest.csv to add labels)
# Or use annotation tool

# 3. Build embeddings
python src/expression_recognition/data/build_embeddings.py \
  --crops data/crops \
  --manifest data/crops/day1/manifest.csv \
  --out embeddings/frame_table.parquet

# 4. Create sequences
python src/expression_recognition/data/make_sequences.py \
  --frames embeddings/frame_table.parquet \
  --stats embeddings/feature_stats.npz \
  --labels embeddings/labels.json \
  --use-split-column \
  --T 12 --stride 4 \
  --out-train embeddings/train_T12.npz \
  --out-val embeddings/val_T12.npz

# 5. Train model
python src/expression_recognition/training/train_gru_embed.py \
  --train embeddings/train_T12.npz \
  --val embeddings/val_T12.npz \
  --epochs 50 --batch-size 32 \
  --outdir models/gru_embed \
  --export-onnx

# 6. Model ready at: models/gru_embed/exported/gru_embed.onnx
```

## Troubleshooting

### InsightFace Installation Issues
```bash
pip install insightface --no-binary insightface
```

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use CPU for embedding extraction: `--device cpu`
- Process video in chunks

### Low Accuracy
- Collect more data (aim for >500 samples per class)
- Increase sequence length: `--T 24`
- Add data augmentation
- Try finetuning: `--finetune-last-block`

## Next Steps

- Implement live inference with webcam
- Add data augmentation for embeddings
- Try different GRU architectures (LSTM, Transformer)
- Multi-task learning (age, gender, expression)

