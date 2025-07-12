import glob
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from lit_gpt.packed_dataset import CombinedDataset, PackedDataset


def create_dataloader(
    *,
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    data_config,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    """Return a dataloader built from the given dataset configuration."""
    datasets = []
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    *,
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path,
    val_data_dir: Optional[Path],
    train_config,
    val_config,
    seed: int = 12345,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders."""
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        data_config=train_config,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            data_config=val_config,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader
