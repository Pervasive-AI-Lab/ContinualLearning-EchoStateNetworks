import torch
import json
import os
import torchaudio
from sys import platform
if platform == 'win32' or platform == 'cygwin': # windows backend
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")
else: # use linux backend by default
    torchaudio.set_audio_backend("sox_io")
from typing import Tuple, Any, Union, Sequence
from torchaudio import transforms
from torch.utils.data import Dataset
from importlib import resources
from clrnn import benchmarks
import torch

CLASS_TO_ID = { 'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5,
                'five': 6, 'four': 7, 'go': 8, 'happy': 9, 'house': 10, 'left': 11,
                'marvel': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17,
                'right': 18, 'seven': 19, 'sheila': 20, 'six': 21, 'stop': 22, 'three': 23,
                'tree': 24, 'two': 25, 'up': 26, 'wow': 27, 'yes': 28, 'zero': 29}

feature_mean = torch.tensor([ 3.0566e-02,  6.0548e-02,  9.4617e-02,  1.1557e-01,  1.4791e-01,
         2.0800e-01,  2.2398e-01,  2.9156e-01,  2.9941e-01,  2.8550e-01,
         2.2858e-01,  1.6175e-01,  1.2211e-01,  9.2752e-02,  5.1397e-02,
         3.6648e-02,  4.0566e-02,  4.5277e-02,  3.0579e-02,  1.6257e-02,
         1.1968e-02,  9.4869e-03,  7.6366e-03,  1.7716e-02,  2.9397e-02,
         2.6692e-02,  1.4608e-02,  1.4320e-02,  1.9941e-02,  1.2594e-02,
         4.5150e-03,  1.4539e-03,  4.5071e-03, -1.0823e-03,  6.5376e-03,
        -2.1014e-04,  1.2609e-03, -5.1994e-03, -6.6572e-03, -5.7101e-03])

feature_std = torch.tensor([0.0764, 0.1775, 0.3704, 0.5256, 0.6108, 0.8428, 0.8531, 1.1451, 1.1738,
        1.1531, 1.0319, 0.7539, 0.6476, 0.5949, 0.3551, 0.2615, 0.2536, 0.2742,
        0.2009, 0.1178, 0.0936, 0.0726, 0.0549, 0.0784, 0.1230, 0.1200, 0.0701,
        0.0525, 0.0762, 0.0658, 0.0272, 0.0143, 0.0176, 0.0181, 0.0106, 0.0053,
        0.0080, 0.0097, 0.0111, 0.0081])



class SSC(Dataset):
        """Synthetic Speech Commands Recognition
        https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset
        This class works with both `Augmented Dataset` and `Augmented Dataset very noisy` version.
        Each audio is preprocessed with Mel Spectrogram. The resulting sequence has the same length
        for all audio files. The length of the sequence depends on the Mel Spectrogram's parameters.

        The test set contains 20% of the entire dataset.

        Args:
            root: folder where audio files are. Audio are grouped in one folder per class.
            classes: list of names of classes to load. None to load all classes. Default to None.
            targets_from_zero: if True, classes will be remapped when necessary to have labels starting from 0
                to n_classes - 1. Only needed when classes is not None.
            split: 'train' or 'test'. Fixed split based on filenames from original dataset.
            n_mels: number of MEL FCSSs feature to create for each audio during preprocessing
            win_length: length of sliding window for Mel Spectrogram in frames
            hop_length: distance between one window and the next in frames
            normalize: if True normalize features with mean 0 and std 1 class-wise. Default False.
            debug: if True loads only a small sample. Use only to quickly debug your code.
        """
        def __init__(self,
                     root: str = '.data',
                     classes: Union[Sequence[str], None] = None,
                     targets_from_zero: bool = True,
                     split: str = 'train',
                     n_mels: int = 40,
                     win_length: int = 25,
                     hop_length: int = 10,
                     normalize: bool = False,
                     debug: bool = False
                     ) -> None:
            super().__init__()

            if n_mels != 40 and normalize:
                raise ValueError("Precomputed Feature Normalization can be applied only with `n_mels=40`.")

            self.root = root
            assert split == 'train' or split == 'test', "Wrong split for SSC."
            self.split = split
            self.normalize = normalize
            self.n_mels = n_mels  # can be used as input size for models
            self.sample_rate = 16000  # each audio must have this sample rate
            self.debug = debug

            win_length = int(self.sample_rate / 1000 * win_length)
            hop_length = int(self.sample_rate / 1000 * hop_length)
            self.mel_spectr = transforms.MelSpectrogram(sample_rate=self.sample_rate,
                win_length=win_length, hop_length=hop_length, n_mels=n_mels)

            # remap class labels to the selected classes
            if classes is not None:
                assert all([el in list(CLASS_TO_ID.keys()) for el in classes]), "Wrong class name for SSC."
                self.classes = classes
            else:
                self.classes = list(CLASS_TO_ID.keys())

            if targets_from_zero and classes is not None:
                self.class_to_id = dict(zip(classes, list(range(len(classes)))))
            else:
                self.class_to_id = CLASS_TO_ID

            self.data = None
            self.targets = None
            self._load_data()  # preprocess data and set data and targets

        def _load_data(self) -> None:
            """
            Load all audio files and associate the corresponding target.
            Also split between train and test following a fixed split.
            """

            # load test split containing, for each class
            # the test filenames
            f = resources.open_text(benchmarks, "scr_test_split.json")
            #with open('clrnn/benchmarks/scr_test_split.json', 'r') as f:
            test_split_dict = json.load(f)
            f.close()

            data = []
            targets = []
            for classname in self.classes:
                files = [el for el in os.listdir(os.path.join(self.root, classname))
                         if el.endswith('.wav')]

                features = []
                for i, f in enumerate(files):
                    # load appropriate files based on fixed split
                    if self.split == 'test' and f not in test_split_dict[classname]:
                        continue
                    elif self.split == 'train' and f in test_split_dict[classname]:
                        continue

                    audio, sample_rate = torchaudio.load(os.path.join(self.root, classname, f))
                    assert sample_rate == self.sample_rate
                    features.append(self.mel_spectr(audio).permute(0, 2, 1))

                    if self.debug and i > 20:
                        break

                data.append(torch.cat(features, dim=0)) # batch-first sequence
                targets.append(torch.ones(data[-1].size(0)).long() * self.class_to_id[classname])

            self.data = torch.cat(data)
            self.targets = torch.cat(targets)

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """

            Args:
                index: numerical index used to select the pattern

            Returns: (audio, target) where
                audio is the preprocessed audio tensor of size [length, n_mels]
                and target is the class index tensor of size []
            """
            if self.normalize:
                return (self.data[index] - feature_mean) / feature_std, self.targets[index]
            else:
                return self.data[index], self.targets[index]

        def __len__(self) -> int:
            return len(self.data)

