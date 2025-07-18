.PHONY: data train test

DATA_DIR ?= data/slim_star_combined
SLIMPAJAMA ?= /path/to/SlimPajama
STARCODER ?= /path/to/starcoderdata
TOKENIZER ?= data/llama
PERCENTAGE ?= 1.0
DEVICES ?= 8

install:
pip install -e .

data:
python scripts/prepare_starcoder.py --source_path $(STARCODER) --tokenizer_path $(TOKENIZER) --destination_path $(DATA_DIR) --split train --percentage $(PERCENTAGE)
python scripts/prepare_slimpajama.py --source_path $(SLIMPAJAMA) --tokenizer_path $(TOKENIZER) --destination_path $(DATA_DIR) --split validation --percentage $(PERCENTAGE)
python scripts/prepare_slimpajama.py --source_path $(SLIMPAJAMA) --tokenizer_path $(TOKENIZER) --destination_path $(DATA_DIR) --split train --percentage $(PERCENTAGE)

train:
lightning run model --accelerator=cuda --devices=$(DEVICES) pretrain/leoleg.py --devices $(DEVICES) --train_data_dir $(DATA_DIR) --val_data_dir $(DATA_DIR)

test:
bash scripts/setup_test_env.sh
pytest
