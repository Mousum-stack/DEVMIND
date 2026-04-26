#!/bin/bash
# Fix dependency conflicts for fine-tuning

echo "🔧 Fixing environment for fine-tuning..."

# Remove problematic packages
pip uninstall -y keras tensorflow tf-keras 2>/dev/null || true

# Downgrade protobuf to compatible version
pip install 'protobuf==3.20.0' --quiet

# Install/upgrade PyTorch packages with compatible versions
pip install --quiet \
  'torch>=2.0.0' \
  'transformers>=4.39.0,<4.50' \
  'peft>=0.7.0' \
  'datasets>=2.10.0' \
  'numpy<2'

echo "✅ Environment fixed!"
echo "Run: python models/finetune.py"
