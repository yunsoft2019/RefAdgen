import torch
from data_provider.data_set import GenDataset, collate_fn


def get_loader(accelerator, tokenizer, args):
    dataset = GenDataset(
        args.dataset_json_path,
        tokenizer,
        image_root_path=args.train_data_path
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=4,
    )
    return train_dataloader
