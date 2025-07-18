FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0
# build xformers
RUN git clone https://github.com/facebookresearch/xformers.git && \
    cd xformers && git checkout 7a7f2b8 && pip install -v -U . && cd .. && rm -rf xformers
# build flash-attention
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && git checkout 7a983df74215e035e566e37125b0a71e3618f39d && \
    python setup.py install && cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && cd ../xentropy && pip install . && cd ../.. && rm -rf flash-attention
COPY . /app
RUN pip install -e . && pip install -r requirements.txt
CMD ["python", "-m", "lit_gpt"]
