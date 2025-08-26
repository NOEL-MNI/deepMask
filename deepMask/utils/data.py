import os
import numpy as np
import torch
import warnings

from nibabel import load as load_nii
from skimage import transform as skt
from torch.utils.data import Dataset

# ignore warnings
warnings.filterwarnings("ignore")


class InferMaskDataset(Dataset):
    def __init__(self, id, t1, flair, root_dir, transform=None):
        """
        ``torch.utils.data.Dataset`` - abstract class representing a dataset
        inputs:
        - csv_file (string): path to the csv file with annotations
        - root_dir (string): directory with all the images
        - transform (callable, optional): transform to be applied on a sample

        outputs:
        - sample (dict):
            - sample['t1w'] (numpy array): T1-weighted image
            - sample['t2w] (numpy array): T2-weighted FLAIR image
            - sample['filename'] (string): filename
            - sample['id'] (string): patient ID
        """
        self.id = id
        self.t1 = t1
        self.t2 = flair
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return int(1)

    def normalize(image):
        normalized_image = (
        image.astype(dtype=np.float32) - image[np.nonzero(image)].mean()
        ) / image[np.nonzero(image)].std()
        return normalized_image

    def __getitem__(self, idx):
        t1w_fname = os.path.join(self.root_dir, self.t1)
        t2w_fname = os.path.join(self.root_dir, self.t2)

        t1w, t2w = load_nii(t1w_fname).get_fdata(), load_nii(t2w_fname).get_fdata()
        t1w, t2w = normalize(t1w), normalize(t2w)

        sample = {"t1w": t1w, "t2w": t2w, "filename": t1w_fname, "id": self.id}

        if self.transform:
            sample = self.transform(sample)

        return sample


class InferResize(object):
    """
    center crop the image in a sample to a specified size

    input:
    - output_size (tuple or int): desired output size
            if tuple: output is matched to output_size
            if int: smaller of image edges is matched
            to `output_size` preserving the aspect ratio
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        t1w, t2w, t1w_fname, case_id = (
            sample["t1w"],
            sample["t2w"],
            sample["filename"],
            sample["id"],
        )
        t1w = skt.resize(t1w, self.output_size, mode="constant", preserve_range=1)
        t2w = skt.resize(t2w, self.output_size, mode="constant", preserve_range=1)

        return {"t1w": t1w, "t2w": t2w, "filename": t1w_fname, "id": case_id}


class ToTensorInfer(object):
    """
    stacks individual modalities (T1 and FLAIR: ndarray) into
    a multi-dimensional tensor

    inputs:
    - sample (dict):
        - sample['t1w'] (numpy array): T1-weighted image
        - sample['t2w] (numpy array): T2-weighted FLAIR image
        - sample['filename'] (string): filename
        - sample['id'] (string): patient ID

    outputs:
    - sample (dict):
        - sample['image'] (tensor): stacked T1-weighted and FLAIR image
        - sample['filename'] (string): filename
        - sample['id'] (string): patient ID

    """

    def __call__(self, sample):
        t1w, t2w, t1w_fname, case_id = (
            sample["t1w"],
            sample["t2w"],
            sample["filename"],
            sample["id"],
        )

        image = np.stack((t1w, t2w), axis=0)

        return {"image": torch.from_numpy(image), "filename": t1w_fname, "id": case_id}