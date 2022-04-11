"""
Tensorflow based augmented simulators
"""
import pathlib
from typing import Union
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

from . import AugmentedSimulator
from ..dataset import DataSet
from ..dataset import Scaler
from ..logger import CustomLogger

class TensorflowSimulator(AugmentedSimulator):
    """_summary_

        Parameters
        ----------
        model : Union[Model, Sequential]
            _description_
        name : str, optional
            _description_, by default None
        config : ConfigManager
            _description_
        """
    def __init__(self,
                 model: keras.Model,
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

        # optimizer
        if "optimizer" in kwargs:
            if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizers"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizers"]["params"]["lr"])


        # history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}

        # scaler class
        self.scaler = None

        self.predict_time = 0

    def _build_model(self, **kwargs) -> keras.Model:
        """build tensorflow model

        Parameters
        ----------
        **kwargs : dict
            if parameters indicated, it will replace config parameters

        Returns
        -------
        keras.Model
            _description_
        """
        model_ = self.model(**kwargs)
        self.params.update(model_.params)
        return model_._model

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
            #TODO: a callback for tensorboard and another for saving the model
        """
        self.params.update(kwargs)
        processed_x, processed_y = self._process_all_dataset(train_dataset, scaler=scaler, training=True)
        if val_dataset is not None:
            processed_x_val, processed_y_val = self._process_all_dataset(val_dataset, scaler=scaler, training=False)

        self._model.compile(optimizer=self._optimizer,
                            loss=self.params["loss"]["name"],
                            metrics=self.params["metrics"])

        if val_dataset is not None:
            validation_data = (processed_x_val, processed_y_val)
        else:
            validation_data = None

        self.logger.info("Training of {%s} started", self.name)
        history_callback = self._model.fit(x=processed_x,
                                           y=processed_y,
                                           validation_data=validation_data,
                                           epochs=self.params["epochs"],
                                           batch_size=self.params["train_batch_size"],
                                           shuffle=self.params["shuffle"])
        self.logger.info("Training of {%s} finished", self.name)
        self.write_history(history_callback)
        if save_path is not None:
            self.save(save_path)
        return history_callback

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

        processed_x, processed_y = self._process_all_dataset(dataset, scaler=scaler, training=False)

        # make the predictions
        _beg = time.time()
        tmp_res_y = self._model.predict(processed_x, batch_size=self.params["eval_batch_size"])
        self.predict_time = time.time() - _beg

        if scaler is not None:
            observations = scaler.inverse_transform(processed_y)
            predictions = scaler.inverse_transform(tmp_res_y)

        predictions = dataset.reconstruct_output(predictions)
        observations = dataset.reconstruct_output(observations)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.data

        return predictions


    def _process_all_dataset(self, dataset: DataSet, scaler: Scaler=None, training: bool=False) -> tuple:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        Scaler : bool, optional
            _description_, by default True
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        if training:
            extract_x, extract_y = dataset.extract_data()
            if scaler is not None:
                self.scaler = scaler()
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            extract_x, extract_y = dataset.extract_data()
            if dataset._size_x is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the input vector. Either train it "
                                "or load its meta data properly.")
            if dataset._size_y is None:
                raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                                "or load its meta data properly.")
            if scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        return extract_x, extract_y

    def save(self, path: str):
        pass

    def restore(self, path: str):
        pass

    def save_metadata(self, path: str):
        pass

    def load_metadata(self, path: str):
        pass

    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model.summary())

    def plot_model(self, path: str):
        """Plot the model architecture using GraphViz Library

        """
        # verify if GraphViz and pydot are installed
        try:
            import pydot
            import graphviz
        except ImportError as err:
            raise RuntimeError("pydot and graphviz are required to use this function") from err

        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(
            self._model,
            to_file="model.png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=56,
            layer_range=None,
            show_layer_activations=False,
        )

    def write_history(self, history_callback):
        """write the history of the training

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        self.train_losses = history_callback.history["loss"]
        self.val_losses = history_callback.history["val_loss"]

        for metrics in self.params["metrics"]:
            self.train_metrics["train_" + metrics] = history_callback.history[metrics]
            self.val_metrics["val_" + metrics] = history_callback.history["val_" + metrics]

    def count_parameters(self):
        """count the number of parameters of the model

        Returns
        -------
        int
            the number of parameters
        """
        return self._model.count_params()

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
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)