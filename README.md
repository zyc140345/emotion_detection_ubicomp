# Emotion Detection from Text - UbiComp Final Project
**Author**: 张易诚 | 2025-05-31

This repository trains, evaluates and visualises a state-of-the-art
Transformer model for multi-class emotion classification on the Kaggle
"[Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)" dataset.

## Core features
1. **Modern backbone** - defaults to `microsoft/deberta-v3-base`.
2. **Parameter-efficient fine-tuning** – optional LoRA (Low-Rank Adaptation) for fast, memory-efficient training.
3. **Advanced losses** - supports Cross-Entropy with optional class weights *or*
   Focal-Loss with optional label smoothing to combat class imbalance.
4. **Advanced sampling** - over-sampling of under-represented classes.
5. **Adversarial training** - Fast Gradient Method (FGM) switch improves robustness.
6. **Early-stopping & LR scheduling** - via 🤗 *Trainer* callbacks.
7. **Rich logging** - metrics saved as JSON & plots (confusion matrix, learning-curves) saved as PNG.
8. **Reproducibility** - full training args + git SHA (when available) emitted to `runs/` folder.

## Quick start

```bash
# (1) Install deps
pip install -r requirements.txt

# (2) Place the Kaggle csv in ./tweet_emotions.csv.

# (3) Train & evaluate
python train.py \
    --dataset_path ./tweet_emotions.csv \
    --model_name microsoft/deberta-v3-base \
    --max_epochs 10 --batch_size 32 --fp16 --over_sampling \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

# (4) Find artefacts in ./runs/TIMESTAMP_*  (config.json,metrics.json, confusion_matrix.png, curves.png, best_model/)
```