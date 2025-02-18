# Deep clustering

# Paper: "A noise-robust deep clustering of biomolecular ions improves interpretability of mass spectrometric images" by Dan Guo,
# Melanie Christine Föll, Kylie Ariel Bemis, Olga Vitek. 2023, Bioinformatics, DOI: 10.1093/bioinformatics/btad067.

# Original code by Dan Guo: https://github.com/DanGuo1223/mzClustering
# Improved code by Tim Daniel Rose: https://github.com/tdrose/deep_mzClustering
# Our code is adapted from Tim Daniel Rose

from random import sample

import numpy as np
import torch
import torch.nn.functional as functional

from .CAE import CAE
from .cnnClust import cnnClust
from .pseudo_labeling import pseudo_labeling, run_knn


class Deep_Clustering:
    def __init__(
        self,
        ims_dataset,
        acquisition_mask,
        image_shape,
        num_cluster=5,
        label_path=None,
        lr=0.0001,
        batch_size=128,
        knn=True,
        k=10,
        use_gpu=True,
        random_seed=0,
    ):

        super(Deep_Clustering, self).__init__()

        self.label_path = label_path
        self.num_cluster = num_cluster
        self.height = image_shape[0]
        self.width = image_shape[1]
        self.lr = lr
        self.batch_size = batch_size
        self.KNN = knn
        self.k = k
        self.knn_adj = None
        self.loss_func = torch.nn.MSELoss()
        self.use_gpu = use_gpu
        self.image_label = None

        self.random_seed = random_seed
        self.device = torch.device("cuda" if use_gpu else "cpu")

        # Reshaping vectorized images
        images = []
        for index in range(ims_dataset.shape[1]):
            ion_image_flat = ims_dataset[:, index]
            ion_image = self.reshape_image(ion_image_flat, np.invert(acquisition_mask))
            images.append(ion_image)
        self.image_data = np.array(images)
        self.sampleN = len(self.image_data)

        if self.label_path:
            self.label = np.genfromtxt(self.label_path, delimiter=" ")
            self.image_label = np.asarray(self.label, dtype=np.int32)

        # Image normalization
        for i in range(0, self.sampleN):
            current_min = np.min(self.image_data[i, ::])
            current_max = np.max(self.image_data[i, ::])
            self.image_data[i, ::] = (current_max - self.image_data[i, ::]) / (current_max - current_min)

        if knn:
            self.knn_adj = run_knn(self.image_data.reshape((self.image_data.shape[0], -1)), k=self.k)

    @staticmethod
    def get_batch(train_image, batch_size, train_label=None):
        sample_id = sample(range(len(train_image)), batch_size)
        batch_image = train_image[sample_id,]
        if train_label is None:
            batch_label = None
        else:
            batch_label = train_label[sample_id,]
        return batch_image, batch_label, sample_id

    @staticmethod
    def get_batch_sequential(train_image, train_label, batch_size, i):
        if i < len(train_image) // batch_size:
            batch_image = train_image[(batch_size * i) : (batch_size * (i + 1)), :]
            batch_label = train_label[(batch_size * i) : (batch_size * (i + 1))]
        else:
            batch_image = train_image[(batch_size * i) : len(train_image), :]
            batch_label = train_label[(batch_size * i) : len(train_image)]
        return batch_image, batch_label

    def train(self):

        cae = CAE(train_mode=True, height=self.height, width=self.width).to(self.device)
        clust = cnnClust(num_clust=self.num_cluster, height=self.height, width=self.width).to(self.device)

        model_params = list(cae.parameters()) + list(clust.parameters())
        optimizer = torch.optim.RMSprop(params=model_params, lr=0.001, weight_decay=0)
        # torch.optim.Adam(model_params, lr=lr)

        uu = 98
        ll = 46
        loss_list = list()

        torch.manual_seed(self.random_seed)
        if self.use_gpu:
            torch.cuda.manual_seed(self.random_seed)
            torch.backends.cudnn.deterministic = True

        # Pretraining of CAE only
        for epoch in range(0, 11):
            losses = list()
            for it in range(501):

                train_x, train_y, index = self.get_batch(self.image_data, self.batch_size, train_label=self.image_label)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))
                optimizer.zero_grad()
                x_p = cae(train_x)

                loss = self.loss_func(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(f"Pretraining Epoch: {epoch} Loss: {sum(losses) / len(losses):.6f}")

        optimizer = torch.optim.RMSprop(params=model_params, lr=0.01, weight_decay=0.0)

        # Full model training
        for epoch in range(0, 11):

            losses = list()
            losses2 = list()

            train_x, train_y, index = self.get_batch(self.image_data, self.batch_size, train_label=self.image_label)

            train_x = torch.Tensor(train_x).to(self.device)
            train_x = train_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(train_x)
            features = clust(x_p)
            # Normalization of clustering features
            features = functional.normalize(features, p=2, dim=-1)
            # Another normalization !?
            features = features / features.norm(dim=1)[:, None]
            # Similarity as defined in formula 2 of the paper
            sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

            for it in range(31):

                train_x, train_y, index = self.get_batch(self.image_data, self.batch_size, train_label=self.image_label)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))

                optimizer.zero_grad()
                x_p = cae(train_x)

                loss1 = self.loss_func(x_p, train_x)

                features = clust(x_p)
                # Feature normalization
                features = functional.normalize(features, p=2, dim=-1)
                features = features / features.norm(dim=1)[:, None]
                # Similarity computation as defined in formula 2 of the paper
                sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

                sim_numpy = sim_mat.cpu().detach().numpy()
                # Get all sim values from the batch excluding the diagonal

                tmp2 = [
                    sim_numpy[i][j] for i in range(0, self.batch_size - 1) for j in range(self.batch_size - 1) if i != j
                ]
                # Compute upper and lower percentiles according to uu & ll
                ub = np.percentile(tmp2, uu)
                lb = np.percentile(tmp2, ll)

                pos_loc, neg_loc = pseudo_labeling(
                    ub=ub, lb=lb, sim=sim_numpy, index=index, knn=self.KNN, knn_adj=self.knn_adj
                )
                pos_loc = pos_loc.to(self.device)
                neg_loc = neg_loc.to(self.device)

                pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
                neg_entropy = torch.mul(-torch.log(torch.clip(1 - sim_mat, 1e-10, 1)), neg_loc)

                loss2 = pos_entropy.sum() / pos_loc.sum() + neg_entropy.sum() / neg_loc.sum()

                loss = 1000 * loss1 + loss2

                losses.append(loss1.item())
                losses2.append(loss2.item())
                loss.backward()
                optimizer.step()
                loss_list.append(sum(losses) / len(losses))

            uu = uu - 1
            ll = ll + 4
            print(f"Training Epoch: {epoch} Loss: {sum(losses) / len(losses):.6f}")
        return cae, clust

    def inference(self, cae, clust):
        with torch.no_grad():
            pred_label = list()

            test_x = torch.Tensor(self.image_data).to(self.device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(test_x)
            psuedo_label = clust(x_p)

            psuedo_label = torch.argmax(psuedo_label, dim=1)
            pred_label.extend(psuedo_label.cpu().detach().numpy())
            pred_label = np.array(pred_label)

            return pred_label

    def reshape_image(self, data, background_mask):
        # Fill background pixels with zeros (default) or NaN
        pixel_grid = np.zeros((self.height * self.width,))
        pixel_grid[np.invert(background_mask)] = data

        # Reshape data
        image = np.reshape(pixel_grid, [self.height, self.width])
        return image
