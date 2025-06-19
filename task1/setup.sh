#!/bin/bash

# 安装依赖
pip install torch torchvision torchaudio
pip install transformers tqdm matplotlib scikit-learn pandas numpy
pip install huggingface_hub datasets accelerate

export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download bert-base-uncased --local-dir bert-base-uncased --local-dir-use-symlinks False

mkdir -p outputs/plots

echo "依赖安装完成，可以运行模型。"

