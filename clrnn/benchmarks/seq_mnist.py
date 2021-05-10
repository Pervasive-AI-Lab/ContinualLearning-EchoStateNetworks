from typing import Tuple, Any

from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor


class SequentialMNIST(MNIST):
    """ Sequential MNIST Dataset.
        A classic benchmark for recurrent neural networks.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(
            self,
            root: str = '.data',
            train: bool = True,
            download: bool = True,
            pixel_per_step: int = 1
    ) -> None:
        super().__init__(root=root, train=train, download=download)
        self.pixel_per_step = pixel_per_step
        assert 784 % pixel_per_step == 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (flat_image, target) where flat_image is the 1D flatten of
             the image and target is the index of the target class.
        """
        img, target = super().__getitem__(index)
        img = to_tensor(img).reshape(-1, self.pixel_per_step)
        return img, target

    def __len__(self) -> int:
        return len(self.data)
