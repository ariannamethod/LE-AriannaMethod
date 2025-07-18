#!/usr/bin/env bash
# Build flash-attention and install Python dependencies for LE.
set -e

# Clone flash-attention pinned commit
if [ ! -d flash-attention ]; then
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention
    git checkout 7a983df74215e035e566e37125b0a71e3618f39d
    python setup.py install
    cd csrc/rotary && pip install .
    cd ../layer_norm && pip install .
    cd ../xentropy && pip install .
    cd ../..
    rm -rf flash-attention
fi

pip install -r requirements.txt tokenizers sentencepiece
pip install -e .
pip install -r requirements-dev.txt

