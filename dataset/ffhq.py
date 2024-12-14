import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from pathlib import Path
from torchvision.transforms import v2


class FFHQDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.dataset = load_from_disk(root_dir)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        return self.transform(self.dataset[index]["image"])

    def __len__(self):
        return len(self.dataset)


def prepare_dataset(output_dir):
    ds = load_dataset("Dmini/FFHQ-64x64")
    ds = ds["train"].train_test_split(test_size=10_000, shuffle=True, seed=42)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    for key in ["train", "test"]:
        ds[key].remove_columns(["label"])
        ds[key].save_to_disk(output_dir / key, num_shards=1024, num_proc=100)


if __name__ == "__main__":
    output_dir = Path(__file__).parents[2].resolve() / "datasets" / "ffhq"
    prepare_dataset(output_dir)
