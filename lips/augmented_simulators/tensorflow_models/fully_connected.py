"""
Tensorflow fully connected Model
"""
import pathlib
from typing import Union
import json
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow import keras

from ..tensorflow_simulator import TensorflowSimulator
from ...logger import CustomLogger
from ...config import ConfigManager
from ...dataset import DataSet
from ...dataset.scaler import Scaler
from ...utils import NpEncoder

from leap_net.proxy import ProxyLeapNet
from leap_net import ResNetLayer


class TfFullyConnected(TensorflowSimulator):
    """Fully Connected architecture

    Parameters
    ----------
    name : Union[str, None], optional
        _description_, by default None
    scaler : Union[Scaler, None], optional
        _description_, by default None
    bench_config_path : Union[str, pathlib.Path, None], optional
        _description_, by default None
    bench_config_name : Union[str, None], optional
        _description_, by default None
    sim_config_path : Union[str, None], optional
        _description_, by default None
    sim_config_name : Union[str, None], optional
        _description_, by default None
    log_path : Union[None, str], optional
        _description_, by default None

    Raises
    ------
    RuntimeError
        _description_
    """

    def __init__(self,
                 name: Union[str, None] = None,
                 scaler: Union[Scaler, None] = None,
                 bench_config_path: Union[str, pathlib.Path, None] = None,
                 bench_config_name: Union[str, None] = None,
                 sim_config_path: Union[str, None] = None,
                 sim_config_name: Union[str, None] = None,
                 log_path: Union[None, str] = None,
                 **kwargs):
        super().__init__(name=name, log_path=log_path, **kwargs)
        # Benchmark configurations
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        # The config file associoated to this model
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        sim_config_path_default = pathlib.Path(__file__).parent.parent / "configurations" / "tf_fc.ini"
        sim_config_path = sim_config_path if sim_config_path is not None else sim_config_path_default
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        self.name = self.name + '_' + sim_config_name
        # scaler
        self.scaler = scaler() if scaler else None
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)
        # Define layer to be used for the model
        self.layers = {"linear": keras.layers.Dense, "resnet": ResNetLayer}
        self.layer = self.layers[self.params["layer"]]

        # optimizer
        if "optimizer" in kwargs and "name" in kwargs["optimizer"]:
            # TODO : isinstance not working properly on keras, isinstance(keras.optimizers.Optimizer, keras.optimizers.Adam) returns False
            # if not isinstance(kwargs["optimizer"], keras.optimizers.Optimizer):
            #   raise RuntimeError("If an optimizer is provided, it should be a type tensorflow.keras.optimizers")
            self._optimizer = kwargs["optimizer"](self.params["optimizer"]["params"])
        else:
            self._optimizer = keras.optimizers.Adam(learning_rate=self.params["optimizer"]["params"]["lr"])

        self._model: Union[keras.Model, None] = None

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

    def build_model(self):
        """Build the model

        Returns
        -------
        Model
            _description_
        """
        super().build_model()
        input_ = keras.layers.Input(shape=(self.input_size,), name="input")
        x = input_
        x = keras.layers.Dropout(rate=self.params["input_dropout"], name="input_dropout")(x)

        if "scale_input_layer" in self.params and self.params["scale_input_layer"]:
            x = keras.layers.Dense(self.params["layers"][0], name="scaling_input_ResNet")(x)

        for layer_id, layer_size in enumerate(self.params["layers"]):
            x = self.layer(layer_size, name=f"layer_{layer_id}")(x)
            x = keras.layers.Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            x = keras.layers.Dropout(rate=self.params["dropout"], name=f"dropout_{layer_id}")(x)
        output_ = keras.layers.Dense(self.output_size)(x)
        self._model = keras.Model(inputs=input_,
                                  outputs=output_,
                                  name=f"{self.name}_model")
        return self._model

    def process_dataset(self, dataset: DataSet, training: bool = False) -> tuple:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Warning
        -------
        It works with StandardScaler only for the moment.

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

        # extract inputs, line_status, outputs without concatenation
        (inputs, extract_tau), outputs = dataset.extract_data(concat=False)
        line_status = extract_tau[0]

        # extract tau using LeapNetProxy function
        extract_tau = self._extract_tau(dataset)

        # add tau and line_status to inputs
        inputs.extend([extract_tau, line_status])

        # concatenate input features
        inputs = np.concatenate([el.astype(np.float32) for el in inputs], axis=1)

        # concatenate outputs labels
        outputs = np.concatenate([el.astype(np.float32) for el in outputs], axis=1)

        if training:
            # set input and output sizes
            self.input_size = inputs.shape[1]
            self.output_size = outputs.shape[1]
            #TODO : exclude scaling line_status and tau features
            if self.scaler is not None:
                inputs, outputs = self.scaler.fit_transform(inputs, outputs)
        else:
            if self.scaler is not None:
                inputs, outputs = self.scaler.transform(inputs, outputs)

        return inputs, outputs

    #TODO : the process of extracting and transforming tau is the same for the leapNet model, it will be better
    # to migrate them to the parent class
    def _extract_tau(self, dataset: DataSet):
        """ Extract and transform tau according to the processing method defined by the argument `topo_vect_to_tau`

        This function reuses ProxyLeapNet methods to process the tau vector.
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

        """

        # LeapNetProxy initialization
        leap_net_model = ProxyLeapNet(
            attr_x=self.bench_config.get_option("attr_x"),
            attr_y=self.bench_config.get_option("attr_y"),
            attr_tau=self.bench_config.get_option("attr_tau"),
            topo_vect_to_tau=self.params["topo_vect_to_tau"] if "topo_vect_to_tau" in self.params else "raw",
            kwargs_tau=self.params["kwargs_tau"] if "kwargs_tau" in self.params else None,
        )
        # transform a numpy dataset into observations
        obss = self._make_fake_obs(dataset)

        leap_net_model.init(obss)
        extract_tau = [leap_net_model.topo_vect_handler(obs) for obs in obss]

        return np.array(extract_tau)

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

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model

        Parameters
        ----------
        dataset : DataSet
            _description_

        Returns
        -------
        None
            _description_
        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def _post_process(self, dataset, predictions):
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        predictions = super()._post_process(dataset, predictions)
        return predictions

    def _save_metadata(self, path: str):
        super()._save_metadata(path)
        if self.scaler is not None:
            self.scaler.save(path)
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        super()._load_metadata(path)
        if self.scaler is not None:
            self.scaler.load(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]
