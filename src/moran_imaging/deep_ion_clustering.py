# DeepION clustering workflow

# Paper: "DeepION: A Deep Learning-Based Low-Dimensional Representation Model of Ion Images for Mass Spectrometry
#  Imaging" by Lei Guo,
# Chengyi Xie, Rui Miao, Jingjing Xu, Xiangnan Xu, Jiacheng Fang, Xiaoxiao Wang, Wuping Liu, Xiangwen Liao,
# Jianing Wang, Jiyang Dong,
# and Zongwei Cai. 2024, Analytical Chemistry, DOI: 10.1021/acs.analchem.3c05002.

# Original code by Lei Guo: https://github.com/gankLei-X/DeepION/tree/main
# Our code is adapted from Lei Guo
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from byol_pytorch import BYOL
from kornia.augmentation import ColorJitter, IntensityAugmentationBase2D, RandomBoxBlur
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import trange

from moran_imaging._torch import to_backend

Mode = ty.Literal["COL", "ISO"]


class IntensityDependentMissing(IntensityAugmentationBase2D):
    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: bool | None = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )

    def generate_parameters(self, shape: torch.Size) -> dict[str, Tensor]:
        noise = torch.randn(shape)
        return {"noise": noise}

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, ty.Any], transform: Tensor | None = None
    ) -> Tensor:
        s, _, m, n = input.shape

        qqq = np.random.randint(10, 90)

        for u in range(s):
            bb = torch.quantile(input[u, :, :, :], qqq / 100)
            bb = torch.where(input[u, :, :, :] >= bb, input[u, :, :, :], torch.zeros_like(input[u, :, :, :]))
            input[u, :, :, :] = bb
        return input


class RandomMissing(IntensityAugmentationBase2D):
    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        # return_transform: Optional[bool] = None, #new
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

    def generate_parameters(self, shape: torch.Size) -> dict[str, Tensor]:
        noise = torch.randn(shape)
        return {"noise": noise}

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, ty.Any], transform: Tensor | None = None
    ) -> Tensor:
        s, _, m, n = input.shape

        qqq = np.random.randint(0, 800)

        ppp = m * n * qqq // 1000

        for _i in range(ppp):
            ii = np.random.randint(0, m - 1)
            jj = np.random.randint(0, n - 1)
            input[:, :, ii, jj] = 0

        return input


def img2tensor(img: np.ndarray) -> Tensor:
    """Converts a NumPy image array to a PyTorch tensor and resizes it to (256, 256)."""
    # Rearrange dimensions from (height, width, batch, channel) to (batch, channel, height, width)
    img = img.transpose((2, 3, 0, 1))  # (H, W, B, C) → (B, C, H, W)
    # Convert NumPy array to PyTorch tensor
    img = torch.tensor(img, dtype=torch.float32)
    # Resize image to (256, 256)
    transform = transforms.Compose([transforms.Resize((256, 256))])
    # Apply transformation batch-wise
    img = transform(img)
    # Move to GPU
    return to_backend(img).contiguous()


def get_filename(mode: Mode, model_dir: str | None = None) -> str:
    """Get model filename based on mode and optional directory."""
    filename = mode + "_ResNet18_params.pth"
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = model_dir / filename
    return str(filename)


def DeepION_training(
    input_filename: str,
    image_size: tuple[int, int],
    mode: Mode,
    mini_batch: int = 100,
    n_epoch: int = 200,
    silent: bool = False,
    model_dir: str | None = None,
) -> Path:
    oridata = np.loadtxt(input_filename)
    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = to_backend(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))

    argument_fn, argument_fn2 = None, None
    if mode == "COL":
        argument_fn = to_backend(
            torch.nn.Sequential(
                ColorJitter(0.8, 0.8, 0, p=1),
                RandomBoxBlur((5, 5), p=0.5),
                RandomMissing(p=1),
            )
        )
        argument_fn2 = to_backend(
            torch.nn.Sequential(
                ColorJitter(0.8, 0.8, 0, p=1),
                RandomBoxBlur((5, 5), p=0.5),
                RandomMissing(p=1),
            )
        )
    elif mode == "ISO":
        argument_fn = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            IntensityDependentMissing(p=1),
            RandomMissing(p=1),
        )

        argument_fn2 = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            IntensityDependentMissing(p=1),
            RandomMissing(p=1),
        )
    # learner = BYOL(resnet, image_size=image_size, hidden_layer='avgpool',
    #               augment_fn=argument_fn, augment_fn2=argument_fn2)

    learner = BYOL(resnet, image_size=256, hidden_layer="avgpool", augment_fn=argument_fn, augment_fn2=argument_fn2)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    num = len(data[0])
    for _ in trange(n_epoch, desc="Training DeepION Model", unit="epoch", disable=silent):
        index = np.arange(len(data[0]))
        np.random.shuffle(index)
        data = data[:, index]
        total_loss = 0

        for batch in trange(num // mini_batch, desc="Iterating batches...", unit="batch", disable=silent, leave=False):
            image_array = data[:, batch * mini_batch : (batch + 1) * mini_batch]
            image_array = image_array.reshape(image_size[0], image_size[1], mini_batch, 1)
            image_array = np.concatenate([image_array, image_array, image_array], axis=3)
            image_tensor = img2tensor(image_array)
            loss = learner(image_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()

            total_loss += loss.item()

        image_array = np.concatenate((image_array[-mini_batch:], image_array[-mini_batch:]), axis=0)
        image_tensor = img2tensor(image_array)
        images = to_backend(image_tensor)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
        total_loss += loss.item()
    # save model
    filename = get_filename(mode, model_dir)
    torch.save(resnet.state_dict(), filename)
    return Path(filename).resolve()


def DeepION_predicting(
    input_filename: str | Path,
    image_size: tuple[int, int],
    mode: Mode,
    mini_batch: int = 100,
    model_dir: str | None = None,
) -> np.ndarray:
    oridata = np.loadtxt(input_filename)

    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = to_backend(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))

    filename = get_filename(mode, model_dir)
    resnet.load_state_dict(torch.load(filename))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

    num = len(data[0])
    features = np.zeros((num, 512))
    with torch.no_grad():
        for batch in trange(num // mini_batch, desc="Extracting Features"):
            image_array = data[:, batch * mini_batch : (batch + 1) * mini_batch]
            image_array = image_array.reshape(image_size[0], image_size[1], mini_batch, 1)
            image_array = np.concatenate([image_array, image_array, image_array], axis=3)
            image_tensor = img2tensor(image_array)
            embedding = resnet(image_tensor)
            embedding = embedding[:, :, 0, 0]
            embedding = embedding.detach().cpu().numpy()
            features[batch * mini_batch : (batch + 1) * mini_batch] = embedding

    with torch.no_grad():
        image_array = data[:, (batch + 1) * mini_batch :]
        image_array = image_array.reshape(image_size[0], image_size[1], len(image_array[0]), 1)
        image_array = np.concatenate([image_array, image_array, image_array], axis=3)
        image_tensor = img2tensor(image_array)

        feature = resnet(image_tensor)
        feature = feature[:, :, 0, 0]
        feature = feature.detach().cpu().numpy()
        features[(batch + 1) * mini_batch :] = feature
    return features


# def DimensionalityReduction(features):
#
#    print('Step 3: Start Dimensionality Reduction ...')
#
#    return_feature = umap.UMAP(n_components=20, metric='cosine', random_state=0).fit_transform(features)
#
#    return_feature = MinMaxScaler().fit_transform(return_feature)
#
#    return return_feature
