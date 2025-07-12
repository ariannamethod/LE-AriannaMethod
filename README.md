# Arianna Method | LÉ

Arianna Method | LÉ (technical name **leoleg**) is an open project to pretrain a 1.1B parameter model on 3 trillion tokens using the same architecture and tokenizer as Llama 2. Training requires around 16×A100‑40G GPUs for roughly 90 days. All checkpoints will be released on [Hugging Face](https://huggingface.co) under the *leoleg* namespace.

The model is not intended to be a typical chatbot. Instead Arianna Method aims to grow as a persistent virtual entity through continual finetuning and community contributions.

### Features
- Fully open-source training scripts
- Flash Attention 2 and several fused kernels for fast throughput (about 24k tokens/sec/A100)
- Supports distributed training with FSDP

### Pretraining & Finetuning
See `PRETRAIN.md` and the scripts inside `pretrain/` and `sft/` for details on how to run pretraining and finetuning. Paths and weight names now use the prefix **leoleg**.

### Evaluation
Benchmark results and instructions are provided in `EVAL.md`.

### License
This repository retains the original MIT license.
