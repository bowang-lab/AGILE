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

import copy
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
from scipy.stats import pearsonr

from dataset.dataset_test import MolTestDatasetWrapper


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


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            args.config,
            os.path.join(model_checkpoints_folder, "config_finetune.yaml"),
        )


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
    return [col for col in df.columns if col.startswith("desc_")]


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        dir_name = (
            current_time + "_" + config["task_name"] + "_" + config["dataset"]["target"]
        )
        log_dir = os.path.join("finetune", dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        print("Logging to:", log_dir)
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
        __, pred = model(data)  # [N,C]

        if self.normalizer:
            loss = self.criterion(pred, self.normalizer.norm(data.y))
        else:
            loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None

        from models.agile_finetune import AGILE
        model = AGILE(self.config["dataset"]["task"], **self.config["model"]).to(
            self.device
        )
        model = self._load_pre_trained_weights(model)

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

        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": self.config["init_base_lr"]},
                {"params": params},
            ],
            self.config["init_lr"],
            weight_decay=eval(self.config["weight_decay"]),
        )

        if apex_support and self.config["fp16_precision"]:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_corr = -1

        for epoch_counter in range(self.config["epochs"]):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, bn)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)
                    print("Epoch:", epoch_counter, "Iteration:", bn, "Train loss:",loss.item())

                if apex_support and self.config["fp16_precision"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                if self.config["dataset"]["task"] == "regression":
                    valid_loss, valid_rgr, valid_corr = self._validate(
                        model, valid_loader
                    )
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        torch.save(
                            model.state_dict(),
                            os.path.join(model_checkpoints_folder, "model.pth"),
                        )
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch_counter
                        best_valid_loss = valid_loss
                        best_valid_rgr = valid_rgr
                        best_valid_corr = valid_corr


                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                self.writer.add_scalar(
                    "validation_rmse", valid_rgr, global_step=valid_n_iter
                )
                valid_n_iter += 1

        print(
            f"Testing model epoch {best_epoch}, "
            f"best validation RMSE: {best_valid_rgr:.3f}, Corr: {best_valid_corr:.3f}"
        )
        self._test(best_model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./ckpt", self.config["fine_tune_from"], "checkpoints"
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

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config["dataset"]["task"] == "regression":
            predictions = np.array(predictions).flatten()
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            corr = pearsonr(labels, predictions)[0]
            print("Validation loss:", valid_loss, "RMSE:", rmse, "Corr:", corr)
            return valid_loss, rmse, corr

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, "checkpoints", "model.pth")
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config["dataset"]["task"] == "regression":
            predictions = np.array(predictions).flatten()
            labels = np.array(labels)
            self.rmse = mean_squared_error(labels, predictions, squared=False)
            self.corr = pearsonr(labels, predictions)[0]
            print(
                f"Test loss: {test_loss:.4f}, "
                f"Test RMSE: {self.rmse:.3f}, Test Corr: {self.corr:.3f}"
            )

        # save predictions and labels to csv
        pred_to_save = predictions if predictions.ndim == 1 else predictions[:, 1]
        df = pd.DataFrame(
            {
                "predictions": pred_to_save,
                "labels": labels,
                "pred_rank": (-pred_to_save).argsort().argsort() + 1,
                "label_rank": (-labels).argsort().argsort() + 1,
            }
        )
        df.to_csv(os.path.join(self.writer.log_dir, "testset_preds.csv"), index=False)


def main(config):
    dataset = MolTestDatasetWrapper(config["batch_size"], **config["dataset"])

    agile_finetune = FineTune(dataset, config)
    agile_finetune.train()

    if config["dataset"]["task"] == "regression":
        agile_finetune.res = agile_finetune.rmse
    return agile_finetune


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file.")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

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

    else:
        raise ValueError("Undefined fine-tuning task!")

    print(config)

    results_list = []
    for target in target_list:
        config["dataset"]["target"] = target
        finetune_agent = main(config)
        result = finetune_agent.res
        results_list.append([target, result])
        
    print(f"Results saved to {finetune_agent.writer.log_dir}")
