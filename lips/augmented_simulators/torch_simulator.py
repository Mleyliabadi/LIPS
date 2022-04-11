"""
Tensorflow based augmented simulators
"""
import pathlib
from typing import Union
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim
from torch import nn
from torch import Tensor
#from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import TensorDataset, DataLoader

from . import AugmentedSimulator
from .torch_models.utils import LOSSES, OPTIMIZERS
from ..dataset import DataSet
from ..dataset import Scaler
from ..logger import CustomLogger

class TorchSimulator(AugmentedSimulator):
    """Pytorch based simulators

        .. code-block:: python

            from lips.augmented_simulators.torch_simulator import TorchSimulator
            from lips.augmented_simulators.torch_models import TorchFullyConnected

            params = {"input_size": 784, "output_size": 10}
            torch_sim = TorchSimulator(name="torch_fc",
                                       model=TorchFullyConnected,
                                       **params)
        Parameters
        ----------
        model : nn.Module
            _description_
        name : str, optional
            _description_, by default None
        **kwargs : dict
            supplementary parameters for the model
            It should contain input_size and output_size
            # TODO: infer from dataset
            It will replace the configs in the config file

        """
    def __init__(self,
                 model: nn.Module,
                 name: Union[str, None],
                 log_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(name, model)
        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        self.model = model
        self.params = kwargs
        self._model = self._build_model(**kwargs)

        if name is not None:
            self.name = name
        else:
            self.name = self._model.name

        # history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}

        # scaler class
        self.scaler = None

        self.predict_time = 0

    def _build_model(self, **kwargs) -> nn.Module:
        """build torch model

        Parameters
        ----------
        **kwargs : dict
            if parameters indicated, it will replace config parameters

        Returns
        -------
        nn.Module
            a torch model
        """
        model_ = self.model(**kwargs)
        self.params.update(model_.params)
        return model_

    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              scaler: Scaler = None,
              save_path: Union[None, str] = None,
              **kwargs):
        """Function used to train a neural network

        Parameters
        ----------
        train_dataset : DataSet
            training dataset
        val_dataset : Union[None, DataSet], optional
            validation dataset, by default None
        scaler : Scaler, optional
            scaler used to scale the data, by default None
        save_path : Union[None, str], optional
            the path where the trained model should be saved, by default None
        """
        self.params.update(kwargs)
        train_loader = self._process_all_dataset(train_dataset, scaler=scaler, training=True)
        if val_dataset is not None:
            val_loader = self._process_all_dataset(val_dataset, scaler=scaler, training=False)
        optimizer = self._get_optimizer(optimizer=OPTIMIZERS[self.params["optimizer"]["name"]],
                                        **self.params["optimizer"]["params"])
        for metric_ in self.params["metrics"]:
            self.train_metrics[metric_] = list()
            if val_loader is not None:
                self.val_metrics[metric_] = list()

        self.logger.info("Training of {%s} started", self.name)
        #losses, elapsed_time = train_model(self.model, data_loaders=data)
        for epoch in range(1, self.params["epochs"]+1):
            train_loss_epoch, train_metrics_epoch = self._train_one_epoch(epoch, train_loader, optimizer)
            self.train_losses.append(train_loss_epoch)
            for nm_, arr_ in self.train_metrics.items():
                arr_.append(train_metrics_epoch[nm_])

            if val_loader is not None:
                val_loss_epoch, val_metrics_epoch = self._validate(val_loader)
                self.val_losses.append(val_loss_epoch)
                for nm_, arr_ in self.val_metrics.items():
                    arr_.append(val_metrics_epoch[nm_])

            # check point
            if self.params["save_freq"] and (save_path is not None):
                if epoch % self.params["ckpt_freq"] == 0:
                    self.save(save_path, epoch)

        # save the final model
        if save_path:
            self.save(save_path)

    def _train_one_epoch(self, epoch:int, train_loader: DataLoader, optimizer: optim.Optimizer) -> set:
        """
        train the model at a epoch
        """
        self._model.train()
        torch.set_grad_enabled(True)

        total_loss = 0
        metric_dict = dict()

        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        for _, batch_ in enumerate(train_loader):
            if len(batch_) == 2:
                data, target = batch_
                loss_func = self._get_loss_func()
            elif len(batch_) == 3:
                data, target, seq_len = batch_
                loss_func = self._get_loss_func(seq_len)
            else:
                raise NotImplementedError("each batch should contain at most 3 tensors")
            data, target = data.to(self.params["device"]), target.to(self.params["device"])
            optimizer.zero_grad()
            # h_0 = self.model.init_hidden(data.size(0))
            # prediction, _ = self.model(data, h_0)
            prediction = self._model(data)
            loss = loss_func(prediction, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(data)
            for metric in self.params["metrics"]:
                if len(batch_) == 2:
                    metric_func = LOSSES[metric](reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(data)
                    metric_dict[metric] += metric_value
                elif len(batch_) == 3:
                    metric_func = LOSSES[metric](seq_len, self.params["device"], reduction="mean")
                    metric_value = metric_func(prediction, target)
                    metric_value = metric_value.item()*len(data)
                    metric_dict[metric] += metric_value

        mean_loss = total_loss/len(train_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(train_loader.dataset)
        print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])
        return mean_loss, metric_dict

    def _validate(self, val_loader: DataLoader, **kwargs) -> set:
        """function used for validation of the model

        It is separated from evaluate function, because it should be called at each epoch during training

        Parameters
        ----------
        val_loader : DataLoader
            _description_

        Returns
        -------
        set
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.params.update(kwargs)
        self._model.eval()
        total_loss = 0
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        with torch.no_grad():
            for _, batch_ in enumerate(val_loader):
                if len(batch_) == 2:
                    data, target = batch_
                    loss_func = self._get_loss_func()
                elif len(batch_) == 3:
                    data, target, seq_len = batch_
                    loss_func = self._get_loss_func(seq_len)
                else:
                    raise NotImplementedError("each batch should contain at most 3 tensors")
                data, target = data.to(self.params["device"]), target.to(self.params["device"])
                #h_0 = self.model.init_hidden(data.size(0))
                #prediction, _ = self.model(data, h_0)
                prediction = self._model(data)
                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(data)

                for metric in self.params["metrics"]:
                    if len(batch_) == 2:
                        metric_func = LOSSES[metric](reduction="mean")
                        metric_value = metric_func(prediction, target)
                        metric_value = metric_value.item()*len(data)
                        metric_dict[metric] += metric_value
                    elif len(batch_) == 3:
                        metric_func = LOSSES[metric](seq_len, self.params["device"], reduction="mean")
                        metric_value = metric_func(prediction, target)
                        metric_value = metric_value.item()*len(data)
                        metric_dict[metric] += metric_value

        mean_loss = total_loss/len(val_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(val_loader.dataset)
        print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        return mean_loss, metric_dict

    def evaluate(self, dataset: DataSet, scaler: Scaler=None, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        scaler : Scaler, optional
            scaler used for normalization, by default None
        """
        if "batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["batch_size"]
        self.params.update(kwargs)

        if scaler is None:
            scaler = self.scaler
        test_loader = self._process_all_dataset(dataset, scaler=scaler, training=False)
        # activate the evaluation mode
        self._model.eval()
        predictions = []
        observations = []
        total_loss = 0
        metric_dict = dict()
        for metric in self.params["metrics"]:
            metric_dict[metric] = 0

        total_time = 0
        with torch.no_grad():
            for _, batch_ in enumerate(test_loader):
                if len(batch_) == 2:
                    data, target = batch_
                    loss_func = self._get_loss_func()
                elif len(batch_) == 3:
                    data, target, seq_len = batch_
                    loss_func = self._get_loss_func(seq_len)
                else:
                    raise NotImplementedError("each batch should contain at most 3 tensors")
                data, target = data.to(self.params["device"]), target.to(self.params["device"])
                #TODO : for RNN, we need to initialize hidden state, but it should be done inside the model
                #h_0 = self.model.init_hidden(data.size(0))
                #prediction, _ = self.model(data, h_0)
                _beg = time.time()
                prediction = self._model(data)
                total_time += time.time() - _beg
                if scaler is not None:
                    prediction = scaler.inverse_transform(prediction)
                    target = scaler.inverse_transform(target)
                predictions.append(prediction.numpy())
                observations.append(target.numpy())

                loss = loss_func(prediction, target)
                total_loss += loss.item()*len(data)

                for metric in self.params["metrics"]:
                    if len(batch_) == 2:
                        metric_func = LOSSES[metric](reduction="mean")
                        metric_value = metric_func(prediction, target)
                        metric_value = metric_value.item()*len(data)
                        metric_dict[metric] += metric_value
                    elif len(batch_) == 3:
                        metric_func = LOSSES[metric](seq_len, self.params["device"], reduction="mean")
                        metric_value = metric_func(prediction, target)
                        metric_value = metric_value.item()*len(data)
                        metric_dict[metric] += metric_value

        mean_loss = total_loss/len(test_loader.dataset)
        for metric in self.params["metrics"]:
            metric_dict[metric] /= len(test_loader.dataset)
        print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
              [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        predictions = dataset.reconstruct_output(np.concatenate(predictions))
        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.reconstruct_output(np.concatenate(observations))
        self.predict_time = total_time
        return predictions#mean_loss, metric_dict

    def _get_loss_func(self, *args) -> Tensor:
        """
        Helper to get loss
        """
        if len(args) > 0:
            # for Masked RNN loss. args[0] is the list of sequence lengths
            loss_func = LOSSES[self.params["loss"]["name"]](args[0], self.params["device"])
        else:
            loss_func = LOSSES[self.params["loss"]["name"]](**self.params["loss"]["params"])
        return loss_func

    def _get_optimizer(self, optimizer: optim.Optimizer=optim.Adam, **kwargs):
        """get the optimizer

        Parameters
        ----------
        optimizer : optim.Optimizer, optional
            _description_, by default optim.Adam
        **kwargs : dict
            the parameters for optimizer
        Returns
        -------
        _type_
            _description_
        """
        return optimizer(self._model.parameters(), **kwargs)

    def _process_all_dataset(self, dataset: DataSet, scaler: Scaler=None, training: bool=False) -> DataLoader:

        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        scaler : Scaler, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False

        Returns
        -------
        DataLoader
            _description_
        """
        if training:
            batch_size = self.params["train_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if scaler is not None:
                self.scaler = scaler()
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if dataset._size_x is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the input vector. Either train it "
                                "or load its meta data properly.")
            if dataset._size_y is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                                "or load its meta data properly.")
            if scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        torch_dataset = TensorDataset(torch.from_numpy(extract_x).float(), torch.from_numpy(extract_y).float())
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader

    ###############################################
    # function used to save and restore the model #
    ###############################################
    def save(self, path: str, epoch: Union[int, None]=None):
        """Save the model

        Parameters
        ----------
        path : ``str``
            _description_
        epoch : Union[``int``, ``None``], optional
            _description_, by default None
        """
        save_path = pathlib.Path(path)
        if not save_path.exists():
            save_path.mkdir()
        else:
            shutil.rmtree(path)

        epoch = str(epoch) if epoch is not None else "last"

        if epoch is None:
            self._save_metadata()

        file_name = pathlib.Path(path) / self.name + epoch + ".pt"
        torch.save(self._model.state_dict(), file_name)

    def _save_metadata(self):
        """save model's metadata

        #TODO: save Scaler parameters (mean, std)
        #TODO: save dataset infos (sizes, etc)
        """
        pass

    def restore(self, path: str):
        """
        restore the model
        """
        self._model.load_state_dict(torch.load(path))
        return self._model

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        pass


    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model)

    def count_parameters(self):
        """
        count the number of parameters in the model
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = len(self.params["metrics"])
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)
        ax[0].set_title("MSE")
        ax[0].plot(self.train_losses, label='train_loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='val_loss')
        for idx_, metric_name in enumerate(self.params["metrics"]):
            ax[idx_].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(2):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            fig.savefig(save_path)

