from itertools import repeat


def inf_loop(dataloader):
    """wrapper function for endless data loader."""
    for loader in repeat(dataloader):
        yield from loader
