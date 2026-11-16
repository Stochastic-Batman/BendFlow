import numpy as np
from datasets import load_dataset

# name -> (Hugging Face repository, image column, (channels, height, width))
_DATASETS = {
    "mnist": ("ylecun/mnist", "image", (1, 28, 28)),
    "fashion_mnist": ("zalando-datasets/fashion_mnist", "image", (1, 28, 28)),
    "cifar10": ("uoft-cs/cifar10", "img", (3, 32, 32)),
}


# (channels, height, width) of one image
def image_shape(name: str) -> tuple[int, int, int]:
    return _DATASETS[name][2]


# float32 array (N, d) with values in [-1, 1], each row flattened in (channels, height, width) order.
# With return_labels the integer labels are returned as a second array.
def load_images(name: str, split: str = "train", return_labels: bool = False):
    repo, column, shape = _DATASETS[name]
    ds = load_dataset(repo, split=split).with_format("numpy")
    images = np.asarray(ds[column])
    if images.ndim == 4:  # (N, H, W, C) -> (N, C, H, W)
        images = images.transpose(0, 3, 1, 2)
    x = images.reshape(images.shape[0], -1).astype(np.float32) / 127.5 - 1.0
    assert x.shape[1] == int(np.prod(shape)), (x.shape, shape)
    if return_labels:
        return x, np.asarray(ds["label"])
    return x
