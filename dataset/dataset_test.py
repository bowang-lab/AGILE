# Copyright (c) 2023 Shihao Ma, Haotian Cui, WangLab @ U of T

# This source code is modified from https://github.com/yuyangw/MolCLR 
# under MIT License. The original license is included below:
# ========================================================================
# MIT License

# Copyright (c) 2021 Yuyang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import codecs
import os
import io
import csv
from typing import List, Optional
from typing_extensions import Literal
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    print("Number of scaffold sets: %d" % len(scaffold_sets))
    for i, scaffold_set in enumerate(scaffold_sets):
        print("Scaffold set %d: %d molecules" % (i, len(scaffold_set)))

    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    if isinstance(data_path, str):
        csv_file = open(data_path)
    elif isinstance(data_path, io.IOBase):
        csv_file = data_path
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for i, row in enumerate(csv_reader):
        smiles = row["smiles"]
        smiles = codecs.decode(smiles, "unicode_escape")
        label = row[target]
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != "":
            smiles_data.append(smiles)
            if task == "classification":
                labels.append(int(label))
            elif task == "regression":
                labels.append(float(label))
            else:
                ValueError("task must be either regression or classification")
    print(len(smiles_data))
    return smiles_data, labels


def read_cols(data_path, cols, return_np=True):
    """
    Reads a csv file and returns the columns specified in cols.

    Args:
        data_path: path to csv file
        cols: list of column names to return
        return_np: if True, returns a numpy array instead of a list
    """
    df = pd.read_csv(data_path)
    if return_np:
        return df[cols].to_numpy()
    else:
        return df[cols].values.tolist()


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1
        if "qm9" in data_path and target in ["homo", "lumo", "gap", "zpve", "u0"]:
            self.conversion = 27.211386246
            print(target, "Unit conversion needed!")

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == "classification":
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == "regression":
            y = torch.tensor(
                self.labels[index] * self.conversion, dtype=torch.float
            ).view(1, -1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWithAdditionalFeatures(MolTestDataset):
    def __init__(self, data_path, target, task, feature_cols):
        if feature_cols is None or len(feature_cols) == 0:
            raise ValueError(
                "found empty feature_cols, should simply use MolTestDataset"
            )
        super().__init__(data_path, target, task)
        self.feature_cols = read_cols(data_path, feature_cols)

    def __getitem__(self, index):
        data: Data = super().__getitem__(index)
        feature = torch.tensor(self.feature_cols[index], dtype=torch.float)
        data.feat = feature.unsqueeze(0)
        return data


class MolTestDatasetWrapper(object):
    def __init__(
        self,
        batch_size,
        num_workers,
        valid_size,
        test_size,
        data_path,
        target: str,
        task: Literal["classification", "regression"],
        splitting: Literal["scaffold", "random"],
        feature_cols: List[str] = [],
    ):
        """
        Args:
            feature_cols: list of column names for additional graph features
        """
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.feature_cols = feature_cols
        assert splitting in ["random", "scaffold"]

    @property
    def dataset(self):
        if not hasattr(self, "_dataset"):
            if len(self.feature_cols) > 0:
                self._dataset = MolTestDatasetWithAdditionalFeatures(
                    self.data_path, self.target, self.task, self.feature_cols
                )
            else:
                self._dataset = MolTestDataset(
                    data_path=self.data_path, target=self.target, task=self.task
                )
        return self._dataset

    @property
    def all_smiles(self):
        return self.dataset.smiles_data

    def get_data_loaders(self):
        (
            train_loader,
            valid_loader,
            test_loader,
        ) = self.get_train_validation_data_loaders(self.dataset)
        return train_loader, valid_loader, test_loader

    def get_fulldata_loader(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == "random":
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = (
                indices[:split],
                indices[split : split + split2],
                indices[split + split2 :],
            )

        elif self.splitting == "scaffold":
            train_idx, valid_idx, test_idx = scaffold_split(
                train_dataset, self.valid_size, self.test_size
            )

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            drop_last=False,
        )
        valid_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers,
            drop_last=False,
        )

        return train_loader, valid_loader, test_loader
