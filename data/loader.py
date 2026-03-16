from torch.utils.data import DataLoader

from .pfsd import TrajectoryDataset, seq_collate

def data_loader(args, data_split='train', shuffle=True):

    if args.dataset_name == 'pfsd':
        dset = TrajectoryDataset(
            args.dataset_dir,
            data_split=data_split,
            device=args.device)
        seq_col = seq_collate

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_col)
    return dset, loader
