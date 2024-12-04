# Copyright (c) 2023 Shihao Ma, Haotian Cui, WangLab @ U of T

# inference and visualization
from io import StringIO
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from rdkit import Chem
import rdkit.Chem.Draw as Draw
import matplotlib
from matplotlib import pyplot as plt
import umap

from dataset.dataset_test import MolTestDatasetWrapper, MolTestDataset
from torch_geometric.explain import Explainer, GNNExplainer
from utils.plot import _image_scatter, facecolors_customize
from utils.constants import (
    R2_to_type,
    R3_to_type,
    R2_to_chain_length,
    R3_to_chain_length,
)

apex_support = False
try:
    sys.path.append("./apex")
    from apex import amp

    apex_support = True
except:
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False



class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def get_desc_cols(fname):
    """Get the descriptor columns from a csv file."""
    df = pd.read_csv(fname)
    return [
        col
        for col in df.columns
        if col not in ["smiles", "expt_Hela", "expt_Raw", "label", "labels"]
    ]


class Inference(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_dir = os.path.join("finetune", config["model_to_evaluate"])
        self.dataset = dataset
        self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data.x, data.edge_index, data.edge_attr, data.batch, data.feat)  # [N,C]
        # __, pred = model(data)  # [N,C]

        if self.config["dataset"]["task"] == "classification":
            loss = self.criterion(pred, data.y.flatten())
        elif self.config["dataset"]["task"] == "regression":
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def inference(self):
        data_loader = self.dataset.get_fulldata_loader()

        self.normalizer = None

        from models.agile_finetune import AGILE
        model = AGILE(self.config["dataset"]["task"], **self.config["model"]).to(
            self.device
        )
        model = self._load_pre_trained_weights(model)

        self.model = model

        layer_list = []
        for name, param in model.named_parameters():
            if "pred_" in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] not in layer_list, model.named_parameters())
                ),
            )
        )
        
        
        # # save config file
        # _save_config_file(model_checkpoints_folder)

        predictions = []
        embeddings = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(data_loader):
                data = data.to(self.device)

                #emb, pred = model(data.x, data.edge_index, data.edge_attr, data.batch, data.feat)
                emb, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                # import ipdb
                # ipdb.set_trace()

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config["dataset"]["task"] == "classification":
                    pred = F.softmax(pred, dim=-1)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    embeddings.extend(emb.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    embeddings.extend(emb.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config["dataset"]["task"] == "regression":
            predictions = np.array(predictions).flatten()
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            if self.config["task_name"] in ["qm7", "qm8", "qm9"]:
                self.mae = mean_absolute_error(labels, predictions)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
            self.corr = pearsonr(labels, predictions)[0]
            print(
                "Test loss:",
                test_loss,
                "Test RMSE:",
                self.rmse,
                "Test Corr:",
                self.corr,
            )

        elif self.config["dataset"]["task"] == "classification":
            predictions = np.array(predictions)
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print("Test loss:", test_loss, "Test ROC AUC:", self.roc_auc)

        # save predictions and labels to csv
        all_smiles = self.dataset.all_smiles
        pred_to_save = predictions if predictions.ndim == 1 else predictions[:, 1]
        assert len(all_smiles) == len(pred_to_save)
        df = pd.DataFrame(
            {
                "smiles": all_smiles,
                "predictions": pred_to_save,
                "labels": labels,
                "pred_rank": (-pred_to_save).argsort().argsort() + 1,
                "label_rank": (-labels).argsort().argsort() + 1,
            }
        )
        df.to_csv(
            os.path.join(self.log_dir, f"preds_on_{config['task_name']}.csv"),
            index=False,
        )

        return predictions, embeddings, labels

    def interpret(self):
        data_loader = self.dataset.get_fulldata_loader()

        self.normalizer = None
        from models.agile_finetune_captum import AGILE
        model = AGILE(self.config["dataset"]["task"], **self.config["model"]).to(
            self.device
        )
        model = self._load_pre_trained_weights(model)

        self.model = model

        layer_list = []
        for name, param in model.named_parameters():
            if "pred_" in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] not in layer_list, model.named_parameters())
                ),
            )
        )

        ## explain
        from captum.attr import IntegratedGradients

        def custom_forward(feat, x, edge_index, edge_attr, batch):
            #batch = torch.zeros(x.shape[0], dtype=int).to(device)
            x = x.to(torch.int64)

            _, pred = model(x, edge_index, edge_attr, batch, feat)
            return pred

        interpret_method = IntegratedGradients(custom_forward)

        feat_importance = []
        for bn, cur_samples in enumerate(data_loader):
            cur_samples = cur_samples.to(self.device)

            attributions, approximation_error = interpret_method.attribute(cur_samples.feat,
                                                            additional_forward_args=(cur_samples.x, cur_samples.edge_index, cur_samples.edge_attr, cur_samples.batch),
                                                            internal_batch_size=self.config['batch_size'],
                                                            return_convergence_delta=True)

            feat_importance.append(attributions.detach().cpu().numpy())
        
        feat_importance = np.vstack(feat_importance)
        df = pd.DataFrame(data=feat_importance, columns=self.config['dataset']['feature_cols']).to_csv(os.path.join(self.log_dir, f'feat_importance.csv'), index=False)
        

        return attributions


    def visualize(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray = None,
        predictions: np.ndarray = None,
        color_key: str = "labels",
    ) -> matplotlib.figure.Figure:
        """
        Visualize the embeddings with UMAP

        Args:
            embeddings (np.ndarray): Raw embeddings to visualize
            labels (np.ndarray, optional): Labels of the embeddings
            predictions (np.ndarray, optional): Predictions values. Defaults to None.
            color_key (str): Use which field to color the points. Defaults to "labels".

        Returns:
            matplotlib.figure.Figure
        """
        if color_key == "labels":
            color = labels
            legend_name = "Efficiency"
        elif color_key == "predictions":
            color = predictions
            legend_name = "Predicted efficiency"

        if predictions is not None and labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 10))

        # umap visualization of embeddings with hue as labels
        reducer = umap.UMAP(
            n_neighbors=60,
            min_dist=1.0,
            spread=1.0,
            metric="cosine",
            random_state=12,
        )
        embedding = reducer.fit_transform(embeddings)
        self.umap_emb = embedding
        # customize cmap
        # cmap_ = matplotlib.cm.get_cmap("Accent_r")
        # make a cmap using three key colors
        cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list(
            "mycmap",
            # ["#44548c", "#5a448e", "#377280"],
            # ["#5f6da0", "#735fa2", "#518692"],
            # ["#5a448e", "#735fa2", "#cec6e1", "#bed5db", "#377280"],
            # ["#5a448e", "#907fb7", "#709d7a", "#377280"],
            ["#cec6e1", "#bda3cd", "#907fb7", "#735fa2", "#5a448e"],
        )
        im = ax1.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color,
            s=60 * 1000 / len(color),
            cmap=cmap_,
            alpha=0.9,
        )
        # remove ticks and spines
        ax1.set(xticks=[], yticks=[])
        ax1.set_title("UMAP projection of the dataset", fontsize=24)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        cbar = fig.colorbar(
            im,
            ax=ax1,
            # ticks=[i for i in np.linspace(np.min(color), np.max(color), 5)],
            format="%.1f",
            orientation="vertical",
            shrink=0.5,
        )
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(legend_name, rotation=270, fontsize=16, labelpad=20)

        # correlation scatter plot bettwen labels and predictions
        if predictions is not None and labels is not None:
            ax2.scatter(
                labels,
                predictions,
                s=60 * 1000 / len(color),
                alpha=0.7,
            )
            ax2.set_title("Correlation between labels and predictions", fontsize=24)
            ax2.set_xlabel("Labels", fontsize=16)
            ax2.set_ylabel("Predictions", fontsize=16)
            ax2.tick_params(axis="both", which="major", labelsize=16)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_xlim([np.min(labels), np.max(labels)])
            ax2.set_ylim([np.min(predictions), np.max(predictions)])
            ax2.plot(
                [np.min(labels), np.max(labels)],
                [np.min(predictions), np.max(predictions)],
                "r--",
            )

            # write the correlation coefficient on the plot
            corr = pearsonr(labels, predictions)[0]
            ax2.text(
                0.7,
                0.3,
                f"Corr: {corr:.2f}",
                transform=ax2.transAxes,
                verticalalignment="top",
                fontsize=16,
                color="red",
            )

        return fig

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./finetune", self.config["model_to_evaluate"], "checkpoints"
            )
            state_dict = torch.load(
                os.path.join(checkpoints_folder, "model.pth"), map_location=self.device
            )
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Name of the checkpoint")
    args = parser.parse_args()
    # args = parser.parse_args(["Nov05_22-58-34_lnp_expt_Hela"])
    config_file = os.path.join(
        "./finetune", args.checkpoint, "checkpoints", "config_finetune.yaml"
    )
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    config["model_to_evaluate"] = args.checkpoint

    if config["task_name"] == "lnp_hela_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/finetuning_set_smiles_plus_features.csv"
        target_list = ["expt_Hela"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )

    elif config["task_name"] == "lnp_raw_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/finetuning_set_smiles_plus_features.csv"
        target_list = ["expt_Raw"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )

    elif config["task_name"] == "smiles12000_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/candidate_set_smiles_plus_features.csv"
        target_list = ["desc_ABC/10"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )

    else:
        raise ValueError("Undefined downstream task!")

    print(config)

    results_list = []
    for target in target_list:
        config["dataset"]["target"] = target
        # result = main(config)
        # results_list.append([target, result])
        dataset = MolTestDatasetWrapper(config["batch_size"], **config["dataset"])

        infer_agent = Inference(dataset, config)
        pred, embs, labels = infer_agent.inference()

        attribute = infer_agent.interpret()
