import os
import numpy as np
import torch
from datetime import datetime

import torch_geometric.io
import scipy.io
from scipy import sparse
import numpy as np
from torch_geometric.nn import fps, knn_graph
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random

from param import device, device_cpu


def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)


def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def my_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.float32)


def my_long_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.long)


def my_range(start, end, step=1):
    return torch.arange(start=start, end=end, step=step, device=device, dtype=torch.float32)


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)

def save_path(num_model=None):
    if num_model is None:
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    else:
        folder_str = str(num_model)

    folder_path_models = os.path.join(data_folder_out, folder_str)
    return folder_path_models


class Param:
    def __init__(self):
        self.k_array_len = 40
        self.k_min = 6
        self.k_max = 500

        self.k_array = None
        self.compute_karray()

        self.area_preservation = False
        self.fac_spec = 0.25
        self.fac_norm = 0.04

        self.filter_scale = 5e4

        self.status_log = True

    def compute_karray(self):
        self.k_array = [int(n) for n in
                        np.exp(np.linspace(np.log(self.k_min), np.log(self.k_max), self.k_array_len)).tolist()]

    def reset_karray(self):
        param_base = Param()
        self.k_array_len = param_base.k_array_len
        self.k_min = param_base.k_min
        self.k_max = param_base.k_max
        self.compute_karray()

    def from_dict(self, d):
        for key in d:
            if hasattr(self, key):
                self.__setattr__(key, d[key])

    def print_self(self):
        print("parameters: ")
        p_d = self.__dict__
        for k in p_d:
            print(k, ": ", p_d[k], "  ", end='')
        print("")
        
        



def input_to_batch(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT", "normal", "area"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    for attr in ["A"]:
        dict_out[attr] = np.asarray(mat_dict[attr][0].diagonal(), dtype=np.float32)

    return dict_out


def input_to_batch_aorta(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT", "normal", "area"]:
        if mat_dict[attr].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr], dtype=np.float32)

    for attr in ["A"]:
        dict_out[attr] = np.asarray(mat_dict[attr].diagonal(), dtype=np.float32)

    return dict_out


def batch_to_shape(batch):
    shape = Shape(batch["vert"].squeeze().to(device), batch["triv"].squeeze().to(device, torch.long) - 1)

    for attr in ["evecs", "evals", "SHOT", "A", "normal", "area"]:
        setattr(shape, attr, batch[attr].squeeze().to(device))

    shape.compute_xi_()

    return shape


def shape_from_dict(mat_dict):
    shape = Shape(torch.from_numpy(mat_dict["vert"][0].astype(np.float32)).to(device),
                  torch.from_numpy(mat_dict["triv"][0].astype(np.int64)).to(device) - 1)

    for attr in ["evecs", "evals", "normal", "area", "SHOT"]:
        setattr(shape, attr, torch.tensor(mat_dict[attr][0], device=device, dtype=torch.float32))

    for attr in ["A"]:
        mat = mat_dict[attr][0].diagonal()
        setattr(shape, attr, torch.tensor(mat, device=device, dtype=torch.float32))

    shape.compute_xi_()

    return shape


def load_shape_pair(file_load):
    mat_dict = scipy.io.loadmat(file_load)

    print("Loaded file ", file_load, "")

    shape_x = shape_from_dict(mat_dict["X"][0])
    shape_y = shape_from_dict(mat_dict["Y"][0])

    return shape_x, shape_y


def compute_outer_normal(vert, triv, samples):
    edge_1 = torch.index_select(vert, 0, triv[:, 1]) - torch.index_select(vert, 0, triv[:, 0])
    edge_2 = torch.index_select(vert, 0, triv[:, 2]) - torch.index_select(vert, 0, triv[:, 0])

    face_norm = torch.cross(1e4*edge_1, 1e4*edge_2)

    normal = my_zeros(vert.shape)
    for d in range(3):
        normal = torch.index_add(normal, 0, triv[:, d], face_norm)
    normal = normal / (1e-5 + normal.norm(dim=1, keepdim=True))

    return normal[samples, :]


class Shape:
    def __init__(self, vert=None, triv=None):
        self.vert = vert
        self.triv = triv
        self.samples = None
        self.reset_sampling()
        self.neigh = None
        self.neigh_hessian = None
        self.mahal_cov_mat = None
        self.evecs = None
        self.evals = None
        self.A = None
        self.W = None
        self.basisfeatures = None
        self.SHOT = None
        self.normal = None
        self.area = None
        self.xi = None

    def subsample_fps(self, n_vert):
        assert n_vert <= self.vert.shape[0], "you can only subsample to less vertices than before"

        ratio = n_vert / self.vert.shape[0]
        self.samples = fps(self.vert.detach().to(device_cpu), ratio=ratio).to(device)

    def subsample_random(self, n_vert):
        self.samples = my_long_tensor([random.randint(0, self.vert.shape[0]-1) for _ in range(n_vert)])

    def reset_sampling(self):
        self.samples = my_long_tensor(list(range(self.vert.shape[0])))
        self.neigh = None

    def compute_xi_(self):
        if self.evecs is not None and self.A is not None and self.vert is not None:
            self.xi = torch.mm(self.evecs.transpose(0, 1), self.vert * self.A.unsqueeze(1))

    def get_vert(self):
        return self.vert[self.samples, :]

    def get_vert_shape(self):
        return self.get_vert().shape

    def get_triv(self):
        return self.triv

    def get_triv_np(self):
        return self.triv.detach().cpu().numpy()

    def get_vert_np(self):
        return self.vert[self.samples, :].detach().cpu().numpy()

    def get_vert_full_np(self):
        return self.vert.detach().cpu().numpy()

    def get_neigh(self, num_knn=5):
        if self.neigh is None:
            self.compute_neigh(num_knn=num_knn)

        return self.neigh

    def compute_neigh(self, num_knn=5):
        if len(self.samples) == self.vert.shape[0]:
            self._triv_neigh()
        else:
            self._neigh_knn(num_knn=num_knn)

    def _triv_neigh(self):
        print("Compute triv neigh....")

        self.neigh = torch.cat((self.triv[:, [0, 1]], self.triv[:, [0, 2]], self.triv[:, [1, 2]]), 0)

    def _neigh_knn(self, num_knn=5):
        print("Compute knn....")

        vert = self.get_vert().detach()
        self.neigh = knn_graph(vert.to(device_cpu), num_knn, loop=False).transpose(0, 1).to(device)

    def to(self, device):
        self.vert = self.vert.to(device)
        self.triv = self.triv.to(device)
































