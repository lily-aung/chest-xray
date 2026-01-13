from torch.utils.data import DataLoader

def build_dataloader(dataset, batch_size, shuffle, num_workers):
    num_workers = int(num_workers)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=False
    )
