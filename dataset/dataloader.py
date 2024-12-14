from torch.utils.data import ConcatDataset, DataLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate


def get_dataloaders(config):
    dataloaders = {}
    for split, params in OmegaConf.to_container(config).items():
        num_workers = params.get("num_workers", 1)

        if split == "train":
            drop_last = True
            shuffle = True
        else:
            drop_last = False
            shuffle = False

        datasets = []
        for ds in params["datasets"]:
            datasets.append(instantiate(ds))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert "batch_size" in params, "You must provide batch_size for each split"
        bs = params["batch_size"]

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(
            dataset
        ), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        dataloaders[split] = dataloader
    return dataloaders
