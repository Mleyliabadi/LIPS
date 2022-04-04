# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

import os
import time
import json
import copy
import warnings
from typing import Union
import tempfile
import shutil
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.optimizers as tfko

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input

from . import AugmentedSimulator
from ..dataset import DataSet
from ..config import ConfigManager
from ..logger import CustomLogger


class FullyConnectedAS(AugmentedSimulator):
    """Fully Connected neural network (tensorflow)

    Parameters
    ----------
    name : ``str``, optional
        the name of the model, to be used for saving, by default "FullyConnected"
    benchmark_name : ``str``, optional
        the name of the scenario for which the model should be used. It is used
        to restore the configurations from config section, by default "Benchmark1"
    config_path : Union[``str``, ``None``], optional
        The path where the config file is available. If ``None``, the default config
        will be used, by default ``None``
    attr_x : Union[``tuple``, ``None``], optional
        the list of input variables to be declared. If ``None`` the input variables
        are restored from the config file, by default None
    attr_y : Union[``tuple``, ``None``], optional
        The list of output variables to be declared. If ``None`` the output variables
        are restored from config file, by default None
    sizes_layer : ``tuple``, optional
        the number of layers and neurones in each layer, by default (150, 150)
    lr : ``float``, optional
        the learning rate for the optimizer, by default 3e-4
    layer_act : ``str``, optional
        the layers activation, by default "relu"
    optimizer : Union[``Optimizer``, ``None``], optional
        the optimizer used for optimizing network weights, by default None
    loss : ``str``, optional
        The loss criteria used during training procedure, by default "mse"
    log_path : Union[``str``, ``None``], optional
        the path where the logs should be stored, by default None

    Raises
    ------
    RuntimeError
        If an optimizer is provided, it should be a type tensorflow.keras.optimizers
    """
    def __init__(self,
                 name: str = "FullyConnected",
                 benchmark_name: str = "Benchmark1",
                 config_path: Union[str, None] = None,
                 attr_x: Union[tuple, None] = None,
                 attr_y: Union[tuple, None] = None,
                 sizes_layer=(150, 150),
                 lr: float = 3e-4,  # only used if "optimizer" is not None
                 layer: Layer = Dense,
                 layer_act: str = "relu",
                 optimizer: Union[Optimizer, None] = None,
                 loss: str = "mse",  # loss used to train the model
                 batch_size: int = 128,
                 log_path: Union[str, None] = None
                ):
        AugmentedSimulator.__init__(self, name)
        self.config_manager = ConfigManager(benchmark_name, config_path)
        if attr_x is not None:
            self._attr_x = attr_x
        else:
            self._attr_x = self.config_manager.get_option("attr_x") + self.config_manager.get_option("attr_tau")

        if attr_y is not None:
            self._attr_y = attr_y
        else:
            self._attr_y = self.config_manager.get_option("attr_y")
        self.sizes_layer = copy.deepcopy(sizes_layer)
        self._lr = lr
        self.layer = layer
        self.layer_act = layer_act
        self._loss = loss
        self._batch_size = batch_size
        if optimizer is not None:
            if not isinstance(optimizer, Optimizer):
                raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = optimizer
        else:
            self._optimizer = tfko.Adam(learning_rate=self._lr)

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        # this model normalizes the data by dividing by the variance and removing the means, i need to keep them
        self._std_x = None
        self._std_y = None
        self._m_x = None
        self._m_y = None
        # this is the keras "model"
        self._model: Union[Model, None] = None

        self.predict_time = None

        # create a logger instance
        self.logger = CustomLogger(__class__.__name__, log_path).logger

    def init(self, **kwargs):
        """this function will build the neural network"""
        if self._model is not None:
            # model is already initialized
            return
        if self._size_x is None or self._size_y is None:
            raise RuntimeError("Impossible to initialize the model if i don't know its size. Have you called "
                               "`fully_connected.load_metada` or `fully_connected.train` ?")
        self._model = Sequential()
        input_ = Input(shape=(self._size_x,), name="input")

        # now make the model
        previous = input_
        for layer_id, layer_size in enumerate(self.sizes_layer):
            previous = self.layer(layer_size, name=f"layer_{layer_id}")(previous)
            previous = Activation(self.layer_act, name=f"activation_{layer_id}")(previous)
        output_ = Dense(self._size_y)(previous)
        self._model = Model(inputs=input_,
                            outputs=output_,
                            name=f"{self.name}_model")

    def train(self, nb_iter: int,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              **kwargs) -> dict:
        """function used to train the model

        Parameters
        ----------
        nb_iter : ``int``
            number of epochs
        train_dataset : DataSet
            the training dataset used to train the model
        val_dataset : Union[``None``, DataSet], optional
            the validation dataset used to validate the model, by default None
        **kwargs: ``dict``
            supplementary arguments for ``fit`` method of ``tf.keras.Model``

        Returns
        -------
        dict
            history of the tensorflow model (losses)
        """
        self.logger.info("Training of {%s} started", self.name)
        self._observations[train_dataset.name] = train_dataset.data
        # extract the input and output suitable for learning (matrices) from the generic dataset
        processed_x, processed_y = self._process_all_dataset(train_dataset, training=True)
        if val_dataset is not None:
            processed_x_val, processed_y_val = self._process_all_dataset(val_dataset, training=False)

        # create the neural network (now that I know the sizes)
        self.init()

        # 'compile' the keras model (now that it is initialized)
        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss)

        # train the model
        if val_dataset is not None:
            validation_data = (processed_x_val, processed_y_val)
        else:
            validation_data = None
        history_callback = self._model.fit(x=processed_x,
                                           y=processed_y,
                                           validation_data=validation_data,
                                           epochs=nb_iter,
                                           batch_size=self._batch_size,
                                           **kwargs)
        self.logger.info("Training of {%s} finished", self.name)
        # NB in this function we use the high level keras method "fit" to fit the data. It does not stricly
        # uses the `DataSet` interface. For more complicated training loop, one can always use
        # dataset.get_data(indexes) to retrieve the batch of data corresponding to `indexes` and
        # `self.process_dataset` to process the example of this dataset one by one.
        return history_callback

    def evaluate(self, dataset: DataSet, batch_size: int=32, save_values: bool=False) -> dict:
        """evaluate the model on the given dataset

        Parameters
        ----------
        dataset : DataSet
            the dataset used for evaluation of the model
        batch_size : ``int``, optional
            the batch size used during inference phase, by default 32
        save_values : ``bool``, optional
            whether to save the evaluation resutls (predictions), by default False

        Todo
        ----
        TODO : Save the predictions (not implemented yet)

        Returns
        -------
        dict
            the predictions of the model
        """
        # the observations used for evaluation
        tmp_obs = dataset.get_data(np.arange(len(dataset)))
        #for attr_nm in self._attr_y:
        #    self.observations[attr_nm] = tmp_obs =
        self._observations[dataset.name] = tmp_obs

        # process the dataset
        processed_x, _ = self._process_all_dataset(dataset, training=False)

        # make the predictions
        _beg = time.time()
        tmp_res_y = self._model.predict(processed_x, batch_size=batch_size)
        self.predict_time = time.time() - _beg
        # rescale them
        tmp_res_y *= self._std_y
        tmp_res_y += self._m_y

        # and now output data as a dictionary
        predictions = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            predictions[attr_nm] = tmp_res_y[:, prev_:(prev_ + this_var_size)]
            prev_ += this_var_size
        self._predictions[dataset.name] = predictions

        #TODO : save the predictions and observations to files np
        if save_values:
            pass

        return predictions

    def save(self, path_out: str):
        """This saves the weights of the neural network.

        Parameters
        ----------
        path_out : ``str``
            path where the model should be saved

        Raises
        ------
        RuntimeError
            The path does not exist
        RuntimeError
            The model is not initialized, it cannot be saved
        """
        if not os.path.exists(path_out):
            raise RuntimeError(f"The path {path_out} does not exists.")
        if self._model is None:
            raise RuntimeError("The model is not initialized, it cannot be saved")

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            self.logger.info("Model {%s} is saved at {%s}", self.name, full_path_out)

        if self._model is not None:
            # save the weights
            self._model.save(os.path.join(full_path_out, "model.h5"))

    def restore(self, path: str):
        """Restores the model from a saved one

        We first copy the weights file into a temporary directory, and then load from this one. This is avoid
        file corruption in case the model fails to load.

        Parameters
        ----------
        path : ``str``
            path from where the  model should be restored

        Raises
        ------
        RuntimeError
            Impossible to find a saved model at the indicated path
        """
        nm_file = f"model.h5"
        path_weights = os.path.join(path, self.name, nm_file)
        if not os.path.exists(path_weights):
            raise RuntimeError(f"Impossible to find a saved model named {self.name} at {path}")

        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

    def save_metadata(self, path_out: str):
        """This is used to save the meta data of the augmented simulator

        In this case it saves the sizes, the scalers etc.
        The only difficulty here is that i need to serialize, as json, numpy arrays

        Parameters
        ----------
        path_out : ``str``
            path where the model metadata should be saved
        """
        res_json = {"batch_size": int(self._batch_size),
                    "lr": float(self._lr),
                    "layer_act": str(self.layer_act),
                    "_loss": str(self._loss),
                    "_size_x": int(self._size_x),
                    "_size_y": int(self._size_y)}
        for my_attr in ["_sizes_x", "_sizes_y", "_m_x", "_m_y", "_std_x",
                        "_std_y", "sizes_layer", "_attr_x", "_attr_y"]:
            tmp = getattr(self, my_attr)
            fun = lambda x: x
            if isinstance(tmp, np.ndarray):
                if tmp.dtype == int or tmp.dtype == np.int or tmp.dtype == np.int32 or tmp.dtype == np.int64:
                    fun = int
                elif tmp.dtype == float or tmp.dtype == np.float32 or tmp.dtype == np.float64:
                    fun = float
            res_json[my_attr] = [fun(el) for el in tmp]

        full_path_out = os.path.join(path_out, self.name)
        if not os.path.exists(full_path_out):
            os.mkdir(full_path_out)
            self.logger.info("Model {%s} is saved at {%s}", self.name, full_path_out)

        with open(os.path.join(full_path_out, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True)

    def load_metadata(self, path: str):
        """this is used to load the meta parameters from the model

        Parameters
        ----------
        path : ``str``
            path from where the  model should be restored
        """
        full_path = os.path.join(path, self.name)
        with open(os.path.join(full_path, f"metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)

        self._batch_size = int(res_json["batch_size"])
        self._lr = float(res_json["lr"])
        self.layer_act = str(res_json["layer_act"])
        self._loss = str(res_json["_loss"])
        self._attr_x = res_json["_attr_x"]
        self._attr_y = res_json["_attr_y"]
        self.sizes_layer = res_json["sizes_layer"]
        self._size_x = int(res_json["_size_x"])
        self._size_y = int(res_json["_size_y"])
        self._sizes_x = np.array(res_json["_sizes_x"], dtype=int)
        self._sizes_y = np.array(res_json["_sizes_y"], dtype=int)

        self._m_x = np.array(res_json["_m_x"], dtype=np.float32)
        self._m_y = np.array(res_json["_m_y"], dtype=np.float32)
        self._std_x = np.array(res_json["_std_x"], dtype=np.float32)
        self._std_y = np.array(res_json["_std_y"], dtype=np.float32)

    def _process_all_dataset(self, dataset: DataSet, training: bool = False) -> tuple:
        """Process the dataset for neural network

        This function will extract the whole dataset and format it in a way we can train the
        fully connected neural network from it

        if "training" is `True` then it will also computes the scalers:
        - _std_x
        - _std_y
        - _m_x
        - _m_y

        And the size of the dataset self._size_x and self._size_y

        Parameters
        ----------
        dataset : ``DataSet``
            the ``DataSet`` object to process
        training : ``bool``
            If the model is in training mode, by default ``False``

        Returns
        -------
        tuple
            the processed inputs and outputs for the model

        Raises
        ------
        RuntimeError
            Model cannot be used, we don't know the size of the input vector.
        RuntimeError
            Model cannot be used, we don't know the size of the output vector.
        """
        all_data = dataset.get_data(np.arange(len(dataset)))
        if training:
            # init the sizes and everything
            self._sizes_x = np.array([all_data[el].shape[1] for el in self._attr_x], dtype=int)
            self._sizes_y = np.array([all_data[el].shape[1] for el in self._attr_y], dtype=int)
            self._size_x = np.sum(self._sizes_x)
            self._size_y = np.sum(self._sizes_y)

        if self._size_x is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the input vector. Either train it "
                               "or load its meta data properly.")
        if self._size_y is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                               "or load its meta data properly.")

        res_x = np.concatenate([all_data[el].astype(np.float32) for el in self._attr_x], axis=1)
        res_y = np.concatenate([all_data[el].astype(np.float32) for el in self._attr_y], axis=1)

        if training:
            self._m_x = np.mean(res_x, axis=0)
            self._m_y = np.mean(res_y, axis=0)
            self._std_x = np.std(res_x, axis=0)
            self._std_y = np.std(res_y, axis=0)

            # to avoid division by 0.
            self._std_x[np.abs(self._std_x) <= 1e-1] = 1
            self._std_y[np.abs(self._std_y) <= 1e-1] = 1

        if self._m_x is None or self._m_y is None or self._std_x is None or self._std_y is None:
            raise RuntimeError("Model cannot be used, we don't know the size of the output vector. Either train it "
                               "or load its meta data properly.")

        res_x -= self._m_x
        res_x /= self._std_x
        res_y -= self._m_y
        res_y /= self._std_y

        return res_x, res_y

    def data_to_dict(self):
        """
        This functions return the observations and corresponding predictions of the last evaluation
        """
        return self._observations, self._predictions
