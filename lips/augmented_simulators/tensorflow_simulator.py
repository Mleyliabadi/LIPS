"""
Tensorflow based augmented simulators
"""
import os
import pathlib
from typing import Union
import shutil
import time
import json
import tempfile
import importlib
#import pydantic.json

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from . import AugmentedSimulator
from ..utils import NpEncoder
from ..dataset import DataSet
from ..logger import CustomLogger

try:
    from leap_net.proxy import ProxyLeapNet
except ImportError as err:
    raise RuntimeError("You need to install the leap_net package to use this class") from err

class TensorflowSimulator(AugmentedSimulator):
    """_summary_

        Parameters
        ----------
        name : str, optional
            _description_, by default None
        config : ConfigManager
            _description_
        """
    def __init__(self,
                 name: Union[str, None]=None,
                 log_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(name=name, log_path=log_path, **kwargs)
        # logger
        self.logger = CustomLogger(__class__.__name__, self.log_path).logger
        self._optimizer = None

        self.input_size = None
        self.output_size = None

        # setting seeds
        np.random.seed(1)
        tf.random.set_seed(2)


    def build_model(self):
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
        if self.input_size is None or self.output_size is None:
            raise RuntimeError("input_size is not set")


    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              save_path: Union[None, str] = None,
              **kwargs):
        """Function used to train a neural network

        Parameters
        ----------
        train_dataset : DataSet
            training dataset
        val_dataset : Union[None, DataSet], optional
            validation dataset, by default None
        save_path : Union[None, str], optional
            the path where the trained model should be saved, by default None
            #TODO: a callback for tensorboard and another for saving the model
        """
        super().train(train_dataset, val_dataset)
        self.params.update(kwargs)
        processed_x, processed_y = self.process_dataset(train_dataset, training=True)
        if val_dataset is not None:
            processed_x_val, processed_y_val = self.process_dataset(val_dataset, training=False)

        # init the model
        self.build_model()

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
        self.trained = True
        if save_path is not None:
            self.save(save_path)
        return history_callback

    def predict(self, dataset: DataSet, **kwargs) -> dict:
        """_summary_

        Parameters
        ----------
        dataset : DataSet
            test datasets to evaluate
        """
        super().predict(dataset)

        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
        self.params.update(kwargs)

        #processed_x, processed_y = self._process_all_dataset(dataset, training=False)
        processed_x, _ = self.process_dataset(dataset, training=False)

        # make the predictions
        predictions = self._model.predict(processed_x, batch_size=self.params["eval_batch_size"])

        predictions = self._post_process(dataset, predictions)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = dataset.data

        return predictions

    def process_dataset(self, dataset: DataSet, training: bool) -> tuple:
        """process the datasets for training and evaluation

        each augmented simulator requires its owan data preparation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            _description_
        training : bool, optional
            _description_, by default False

        Returns
        -------
        tuple
            the normalized dataset with features and labels
        """
        super().process_dataset(dataset, training)
        inputs, outputs = dataset.extract_data()

        return inputs, outputs

    def _transform_topo_vect(self, dataset: DataSet, training: bool=True):
        """ Extract and transform topo_vect according to the processing method defined by the argument `topo_vect_to_tau`

            This function reuses either the ProxyLeapNet methods to process the tau vector or the
            self ._transform_tau_given_list.
            See https://github.com/BDonnot/leap_net/blob/master/leap_net/proxy/proxyLeapNet.py for more details.

                From the LeapNet documentation :

                    There are multiple ways to process the `tau` vector from the topology of the grid. Some of these
                    different methods have been coded in the LeapNetProxy and are controlled by the `topo_vect_to_tau`
                    argument:

                    1) `topo_vect_to_tau="raw"`: the most straightforward encoding. It transforms the `obs.topo_vect`
                     directly into a `tau` vector of the same dimension with the convention: if obs.topo_vect[i] == 2
                     for a given `i` then `tau[i] = 1` else `tau[i] = 0`. More details are given in
                     the :func:`ProxyLeapNet._raw_topo_vect`, with usage examples on how to create it.
                    2) `topo_vect_to_tau="all"`: it encodes the global topology of the grid by a one hot encoding of the
                     "local topology" of each substation. It first computes all the possible "local topologies" for
                     all the substations of the grid and then assign a number (unique ID) for each of them. The resulting
                     `tau` vector is then the concatenation of the "one hot encoded" ID of the current "local topology"
                     of each substation. More information is given in :func:`ProxyLeapNet._all_topo_encode`
                     with usage examples on how to create it.
                    3) `topo_vect_to_tau="given_list"`: it encodes the topology into a `tau` vector following the same
                     convention as method 2) (`topo_vect_to_tau="all"`) with the difference that it only considers
                     a given list of possible topologies instead of all the topologies of all the substation of the grid.
                     This list should be provided as an input in the `kwargs_tau` argument. If a topology not given
                     is encounter, it is mapped to the reference topology.
                    4) `topo_vect_to_tau="online_list"`: it encodes the topology into a `tau` vector following the same
                     convention as method 2) (`topo_vect_to_tau="all"`) and 3) (`topo_vect_to_tau="given_list"`) but does
                     not require to specify any list of topologies. Instead, each time a new "local topology" is
                     encountered during training, it will be assigned to a new ID. When encountered again, this new
                     ID will be re used. It can store a maximum of different topologies given as `kwargs_tau` argument.
                     If too much topologies have been encountered, the new ones will be encoded as the reference topology.
                Returns
                -------
                topo_vect

                """

        if training:
            # initialize a fake leapNet proxy
            self._fake_leap_net_proxy = ProxyLeapNet(
                attr_x=self.bench_config.get_option("attr_x"),
                attr_y=self.bench_config.get_option("attr_y"),
                attr_tau=self.bench_config.get_option("attr_tau"),
                topo_vect_to_tau=self.params["topo_vect_to_tau"] if "topo_vect_to_tau" in self.params else "raw",
                kwargs_tau=self.params["kwargs_tau"] if "kwargs_tau" in self.params else None,
            )
            obss = self._make_fake_obs(dataset)
            self._fake_leap_net_proxy.init(obss)

        # if option is given_list then use the accelerated function else use the leapnet porxy method
        if "topo_vect_to_tau" in self.params and self.params["topo_vect_to_tau"] == "given_list":
            return self._transform_tau_given_list(dataset.data["topo_vect"], self._fake_leap_net_proxy.subs_index)
        else:
            if not training : obss = self._make_fake_obs(dataset)
            return np.array([self._fake_leap_net_proxy.topo_vect_handler(obs) for obs in obss])

    def _make_fake_obs(self, dataset: DataSet):
        """
        the underlying _leap_net_model requires some 'class' structure to work properly. This convert the
        numpy dataset into these structures.

        Definitely not the most efficient way to process a numpy array...
        """
        all_data = dataset.data
        class FakeObs(object):
            pass

        if "topo_vect" in all_data:
            setattr(FakeObs, "dim_topo", all_data["topo_vect"].shape[1])

        setattr(FakeObs, "n_sub", dataset.env_data["n_sub"])
        setattr(FakeObs, "sub_info", np.array(dataset.env_data["sub_info"]))

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])
        return obss

    def _transform_tau_given_list(self, topo_vect_input, subs_index, with_tf=True):
        """Transform only the tau vector with respect to LeapNet encodings given a list of predefined topological actions
                Parameters
        ----------
        tau : list of raw topology representations (line_status, topo_vect)

        with_tf : transformation using tensorflow or numpy operations

        Returns
        -------
        tau
            list of encoded topology representations (line_status, topo_vect_encoded)
        """
        ##############
        # WARNING: TO DO
        # if we find two topology matches at a same substation, the current code attribute one bit for each
        # But only one should be choosen in the end (we are not in a quantum state, or it does not make sense to combine topologies at a same substation in the encoding here
        # This can happen when there are several lines disconnected at a substation on which we changed the topology, probably in benchmark 3, but probably not in benchmark 1 and 2

        list_topos = []
        sub_length = []
        for topo_action in self.params["kwargs_tau"]:
            topo_vect = np.zeros(topo_vect_input.shape[1], dtype=np.int32)
            sub_id = topo_action[0]
            sub_topo = np.array(topo_action[1])
            sub_index = subs_index[sub_id][0]
            n_elements = len(sub_topo)
            topo_vect[sub_index:sub_index + n_elements] = sub_topo
            list_topos.append(topo_vect)
            sub_length.append(n_elements)

        list_topos = np.array(list_topos)

        # we are here looking for the number of matches for every element of a substation topology in the predefined list for a new topo_vect observation
        # if the count is equal to the number of element, then the predefined topology is present in topo_vect observation
        # in that case, the binary encoding of that predefined topology is equal to 1, otherwise 0

        import time
        start = time.time()
        if with_tf:
            # count the number of disconnected lines for each substation of topologies in the prefdefined list.
            # These lines could have been connected to either bus_bar1 or bus_bar2, we consider it as a match for that element
            line_disconnected_sub = tf.linalg.matmul((list_topos > 0).astype(np.int32),
                                                     (np.transpose(topo_vect_input) < 0).astype(np.int32))

            # we look at the number of elements on bus_bar1 that match, same for the number of elements on bus_bar2
            match_tensor_bus_bar1 = tf.linalg.matmul((list_topos == 1).astype(np.int32),
                                                     (np.transpose(topo_vect_input) == 1).astype(np.int32))
            match_tensor_bus_bar2 = tf.linalg.matmul((list_topos == 2).astype(np.int32),
                                                     (np.transpose(topo_vect_input) == 2).astype(np.int32))

            # the number of matches is equal to the sum of those 3 category of matches
            match_tensor_adjusted = match_tensor_bus_bar1 + match_tensor_bus_bar2 + line_disconnected_sub

            # we see if all elements match by dividing by the number of elements. If this proportion is equal to one, we found a topology match
            normalised_tensor = match_tensor_adjusted / tf.reshape(np.array(sub_length).astype(np.int32), (-1, 1))

        else:  # with_numpy

            line_disconnected_sub = np.matmul((list_topos > 0), 1 * (np.transpose(topo_vect_input) < 0))

            match_tensor_bus_bar1 = np.matmul((list_topos == 1), 1 * (np.transpose(topo_vect_input) == 1))
            match_tensor_bus_bar2 = np.matmul((list_topos == 2), 1 * (np.transpose(topo_vect_input) == 2))

            match_tensor_adjusted = match_tensor_bus_bar1 + match_tensor_bus_bar2 + line_disconnected_sub

            normalised_tensor = match_tensor_adjusted / np.array(sub_length).reshape((-1, 1))

        boolean_match_tensor = np.array(normalised_tensor == 1.0).astype(np.int8)

        duration_matches = time.time() - start

        #############"
        ## do correction if multiple topologies of a same substation have a match on a given state
        # as it does not make sense to combine topologies at a same substation
        start = time.time()
        boolean_match_tensor = self._unicity_tensor_encoding(boolean_match_tensor)

        duration_correction = time.time() - start
        if (duration_correction > duration_matches):
            print("warning, correction time if longer that matches time: maybe something to better optimize there")
        topo_vect_input = np.transpose(boolean_match_tensor)

        return topo_vect_input

    def _unicity_tensor_encoding(self, tensor):
        """
        do correction if multiple topologies of a same substation have a match on a given state
        as it does not make sense to combine topologies at a same substation
        """
        sub_encoding_pos = np.array([topo_action[0] for topo_action in self.params["kwargs_tau"]])

        # in case of multiple matches of topology for a given substation, encode only one of those topologies as an active bit, not several
        def per_col(a):  # to only have one zero per row
            idx = a.argmax(0)
            out = np.zeros_like(a)
            r = np.arange(a.shape[1])
            out[idx, r] = a[idx, r]
            return out

        for sub in set(sub_encoding_pos):
            indices = np.where(sub_encoding_pos == sub)[0]
            if (len(indices) >= 2):
                tensor[indices, :] = per_col(tensor[indices, :])

        return tensor

    def _post_process(self, dataset, predictions):
        """Do some post processing on the predictions

        Parameters
        ----------
        predictions : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return dataset.reconstruct_output(predictions)


    ###############################################
    # function used to save and restore the model #
    ###############################################
    def save(self, path: str, save_metadata: bool=True):
        """_summary_

        Parameters
        ----------
        path : str
            _description_
        save_metadata : bool, optional
            _description_, by default True
        """
        save_path =  pathlib.Path(path) / self.name
        super().save(save_path)

        self._save_model(save_path)

        if save_metadata:
            self._save_metadata(save_path)

        self.logger.info("Model {%s} is saved at {%s}", self.name, save_path)

    def _save_model(self, path: Union[str, pathlib.Path], ext: str=".h5"):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        file_name = path / ("weights" + ext)
        self._model.save_weights(file_name)

    def _save_metadata(self, path: Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        # for json serialization of paths
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.PosixPath] = str
        #pydantic.json.ENCODERS_BY_TYPE[pathlib.WindowsPath] = str
        self._save_losses(path)
        with open((path / "config.json"), "w", encoding="utf-8") as f:
            json.dump(obj=self.params, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def restore(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        full_path = path / self.name
        if not full_path.exists():
            raise FileNotFoundError(f"path {full_path} not found")
        # load the metadata
        self._load_metadata(full_path)
        self._load_model(full_path)

        self.logger.info("Model {%s} is loaded from {%s}", self.name, full_path)

    def _load_model(self, path: str):
        nm_file = "weights.h5"
        path_weights = path / nm_file
        if not path_weights.exists():
            raise FileNotFoundError(f"Weights file {path_weights} not found")
        self.build_model()
        # load the weights
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(path_weights, nm_tmp)
            # load this copy (make sure the proper file is not corrupted even if the loading fails)
            self._model.load_weights(nm_tmp)

    def _load_metadata(self, path: str):
        """
        load the model metadata
        """
        # load scaler parameters
        #self.scaler.load(full_path)
        self._load_losses(path)
        with open((path / "config.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.params.update(res_json)
        return self.params

    def _save_losses(self, path: Union[str, pathlib.Path]):
        """
        save the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        res_losses = {}
        res_losses["train_losses"] = self.train_losses
        res_losses["train_metrics"] = self.train_metrics
        res_losses["val_losses"] = self.val_losses
        res_losses["val_metrics"] = self.val_metrics
        with open((path / "losses.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_losses, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_losses(self, path: Union[str, pathlib.Path]):
        """
        load the losses
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "losses.json"), "r", encoding="utf-8") as f:
            res_losses = json.load(fp=f)
        self.train_losses = res_losses["train_losses"]
        self.train_metrics = res_losses["train_metrics"]
        self.val_losses = res_losses["val_losses"]
        self.val_metrics = res_losses["val_metrics"]

    #########################
    # Some Helper functions #
    #########################
    def summary(self):
        """summary of the model
        """
        print(self._model.summary())

    def plot_model(self, path: Union[str, None]=None, file_name: str="model"):
        """Plot the model architecture using GraphViz Library

        """
        # verify if GraphViz and pydot are installed
        pydot_found = importlib.util.find_spec("pydot")
        graphviz_found = importlib.util.find_spec("graphviz")
        if pydot_found is None or graphviz_found is None:
            raise RuntimeError("pydot and graphviz are required to use this function")

        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(
            self._model,
            to_file=file_name+".png",
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=56,
            layer_range=None,
            show_layer_activations=False,
        )

    def write_history(self, history: dict):
        """write the history of the training

        Parameters
        ----------
        history_callback : keras.callbacks.History
            the history of the training
        """
        self.train_losses = history.history["loss"]
        self.val_losses = history.history["val_loss"]

        for metric in self.params["metrics"]:
            self.train_metrics[metric] = history.history[metric]
            self.val_metrics[metric] = history.history["val_" + metric]

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
            ax[idx_+1].set_title(metric_name)
            ax[idx_+1].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_+1].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)