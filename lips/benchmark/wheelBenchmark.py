"""
Licence:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""

import os
import shutil
import warnings
import copy
from typing import Union
import importlib

from lips.benchmark import Benchmark
# from .utils.powergrid_utils import get_kwargs_simulator_scenario

from lips.evaluation.transport_evaluation import TransportEvaluation
from lips.physical_simulator.getfemSimulator import PhysicalSimulator,GetfemSimulator
from lips.augmented_simulators.augmented_simulator import AugmentedSimulator
from lips.dataset.pneumaticWheelDataSet import SamplerStaticWheelDataSet,QuasiStaticWheelDataSet

class WheelBenchmark(Benchmark):
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[TransportEvaluation, None]=None,
                 log_path: Union[str, None]=None,
                 train_env_seed: int = 1,
                 val_env_seed: int = 2,
                 test_env_seed: int = 3,
                 test_ood_topo_env_seed: int = 4,
                 initial_chronics_id: int = 0,
                 train_actor_seed: int = 5,
                 val_actor_seed: int = 6,
                 test_actor_seed: int = 7,
                 test_ood_topo_actor_seed: int = 8,
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         dataset=None,
                         augmented_simulator=None,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )


class WeightSustainingWheelBenchmark(WheelBenchmark):
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[TransportEvaluation, None]=None,
                 log_path: Union[str, None]=None,
                 train_env_seed: int = 1,
                 val_env_seed: int = 2,
                 test_env_seed: int = 3,
                 test_ood_topo_env_seed: int = 4,
                 initial_chronics_id: int = 0,
                 train_actor_seed: int = 5,
                 val_actor_seed: int = 6,
                 test_actor_seed: int = 7,
                 test_ood_topo_actor_seed: int = 8,
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )

        self.is_loaded=False
        # TODO : it should be reset if the config file is modified on the fly
        if evaluation is None:
            myEval=TransportEvaluation(config_path=config_path,scenario=benchmark_name)
            self.evaluation = myEval.from_benchmark(benchmark=self)

        # print(self.config.get_options_dict())
        self.env_name = self.config.get_option("env_name")
        self.training_simulator = None
        self.val_simulator = None
        self.test_simulator = None
        self.test_ood_topo_simulator = None

        self.training_actor = None
        self.val_actor = None
        self.test_actor = None
        self.test_ood_topo_actor = None

        self.train_env_seed = train_env_seed
        self.val_env_seed = val_env_seed
        self.test_env_seed = test_env_seed
        self.test_ood_topo_env_seed = test_ood_topo_env_seed

        self.train_actor_seed = train_actor_seed
        self.val_actor_seed = val_actor_seed
        self.test_actor_seed = test_actor_seed
        self.test_ood_topo_actor_seed = test_ood_topo_actor_seed

        self.initial_chronics_id = initial_chronics_id
        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x")\
                     +self.config.get_option("attr_y")


        self.train_dataset = SamplerStaticWheelDataSet("train",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self.val_dataset = SamplerStaticWheelDataSet("val",
                                            attr_names=attr_names,
                                            config=self.config,
                                            log_path=log_path
                                            )

        self._test_dataset = SamplerStaticWheelDataSet("test",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = SamplerStaticWheelDataSet("test_ood_topo",
                                                       attr_names=attr_names,
                                                       config=self.config,
                                                       log_path=log_path
                                                       )

        if load_data_set:
            self.load()

    def load(self):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets)
        self.val_dataset.load(path=self.path_datasets)
        self._test_dataset.load(path=self.path_datasets)
        self.is_loaded = True

    def generate(self, nb_sample_train: int, nb_sample_val: int,
                 nb_sample_test: int, nb_sample_test_ood_topo: int):
        """
        generate the different datasets required for the benchmark
        """
        if self.is_loaded:
            self.logger.warning("Previously saved data will be erased by this new generation")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._fills_actor_simulator()
        if os.path.exists(self.path_datasets):
            self.logger.warning("Deleting path %s that might contain previous runs", self.path_datasets)
            shutil.rmtree(self.path_datasets)

        self.logger.info("Creating path %s to save the current data", self.path_datasets)
        os.mkdir(self.path_datasets)

        self.train_dataset.generate(simulator=self.training_simulator,
                                    actor=self.training_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_train
                                    )
        self.val_dataset.generate(simulator=self.val_simulator,
                                  actor=self.val_actor,
                                  path_out=self.path_datasets,
                                  nb_samples=nb_sample_val
                                  )
        self._test_dataset.generate(simulator=self.test_simulator,
                                    actor=self.test_actor,
                                    path_out=self.path_datasets,
                                    nb_samples=nb_sample_test
                                    )

    def evaluate_predictor(self,
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:

        li_dataset = []
        if dataset == "all":
            li_dataset = [self.val_dataset, self._test_dataset, self._test_ood_topo_dataset]
            keys = ["val", "test", "test_ood_topo"]
        elif dataset == "val" or dataset == "val_dataset":
            li_dataset = [self.val_dataset]
            keys = ["val"]
        elif dataset == "test" or dataset == "test_dataset":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood_topo" or dataset == "test_ood_topo_dataset":
            li_dataset = [self._test_ood_topo_dataset]
            keys = ["test_ood_topo"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate predictor function of Benchmark class
            tmp = self._aux_predict_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_predict_on_single_dataset(self,
                                        dataset: SamplerStaticWheelDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:

        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )
        self.augmented_simulator = augmented_simulator
        predictions = self.augmented_simulator.evaluate(dataset)

        return predictions


    def evaluate_simulator_from_predictions(self,
                                            predictions: dict,
                                            observations: dict,
                                            dataset: str = "all",
                                            save_path: Union[str, None]=None,
                                            **kwargs) -> dict:

        li_dataset = []
        if dataset == "all":
            li_dataset = [self.val_dataset, self._test_dataset, self._test_ood_topo_dataset]
            keys = ["val", "test", "test_ood_topo"]
        elif dataset == "val" or dataset == "val_dataset":
            li_dataset = [self.val_dataset]
            keys = ["val"]
        elif dataset == "test" or dataset == "test_dataset":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood_topo" or dataset == "test_ood_topo_dataset":
            li_dataset = [self._test_ood_topo_dataset]
            keys = ["test_ood_topo"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate predictor function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset_from_prediction(dataset=dataset_,
                                                       predictions=predictions[nm_],
                                                       observations=observations[nm_],
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_evaluate_on_single_dataset_from_prediction(self,
                                        dataset: SamplerStaticWheelDataSet,
                                        predictions: dict,
                                        observations: dict,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:
        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = observations
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=observations,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

    def evaluate_simulator(self,
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:
        """evaluate a trained augmented simulator on one or multiple test datasets

        Parameters
        ----------
        dataset : str, optional
            dataset on which the evaluation should be performed, by default "all"
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            An instance of the class augmented simulator, by default None
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        **kwargs: ``dict``
            additional arguments that will be passed to the augmented simulator
        Todo
        ----
        TODO: add active flow in config file

        Returns
        -------
        dict
            the results dictionary

        Raises
        ------
        RuntimeError
            Unknown dataset selected

        """
        self._create_training_simulator()
        li_dataset = []
        if dataset == "all":
            li_dataset = [self.val_dataset, self._test_dataset, self._test_ood_topo_dataset]
            keys = ["val", "test", "test_ood_topo"]
        elif dataset == "val" or dataset == "val_dataset":
            li_dataset = [self.val_dataset]
            keys = ["val"]
        elif dataset == "test" or dataset == "test_dataset":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood_topo" or dataset == "test_ood_topo_dataset":
            li_dataset = [self._test_ood_topo_dataset]
            keys = ["test_ood_topo"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate simulator function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       save_path=save_path,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)
        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: SamplerStaticWheelDataSet,
                                        augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                                        save_path: Union[str, None]=None,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : SamplerStaticWheelDataSet
            the dataset
        augmented_simulator : Union[PhysicalSimulator, AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
        save_path : Union[str, None], optional
            if indicated the evaluation results will be saved to indicated path, by default None

        Returns
        -------
        dict
            the results dictionary
        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )
        self.augmented_simulator = augmented_simulator
        predictions = self.augmented_simulator.evaluate(dataset)

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       save_path=save_path
                                       )
        return res

    def _create_training_simulator(self):
        """
        Initialize the simulator used for training

        """
        if self.training_simulator is None:
            scenario_params=self.config.get_option("env_params")
            self.training_simulator = GetfemSimulator(**scenario_params)

    def _fills_actor_simulator(self):
        """This function is only called when the data are simulated"""
        self._create_training_simulator()

        scenario_params=self.config.get_option("env_params")
        self.val_simulator = GetfemSimulator(**scenario_params)

        self.test_simulator = GetfemSimulator(**scenario_params)

        self.test_ood_topo_simulator = GetfemSimulator(**scenario_params)


class DispRollingWheelBenchmark(WheelBenchmark):
    def __init__(self,
                 benchmark_path: str,
                 config_path: Union[str, None]=None,
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[TransportEvaluation, None]=None,
                 log_path: Union[str, None]=None,
                 train_env_seed: int = 1,
                 val_env_seed: int = 2,
                 test_env_seed: int = 3,
                 test_ood_topo_env_seed: int = 4,
                 initial_chronics_id: int = 0,
                 train_actor_seed: int = 5,
                 val_actor_seed: int = 6,
                 test_actor_seed: int = 7,
                 test_ood_topo_actor_seed: int = 8,
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         evaluation=evaluation,
                         benchmark_path=benchmark_path,
                         log_path=log_path,
                         config_path=config_path
                        )

        self.is_loaded=False
        # TODO : it should be reset if the config file is modified on the fly
        if evaluation is None:
            myEval=TransportEvaluation(config_path=config_path,scenario=benchmark_name)
            self.evaluation = myEval.from_benchmark(benchmark=self)

        self.env_name = self.config.get_option("env_name")
        self.training_simulator = None
        self.val_simulator = None
        self.test_simulator = None
        self.test_ood_topo_simulator = None

        self.training_actor = None
        self.val_actor = None
        self.test_actor = None
        self.test_ood_topo_actor = None

        self.train_env_seed = train_env_seed
        self.val_env_seed = val_env_seed
        self.test_env_seed = test_env_seed
        self.test_ood_topo_env_seed = test_ood_topo_env_seed

        self.train_actor_seed = train_actor_seed
        self.val_actor_seed = val_actor_seed
        self.test_actor_seed = test_actor_seed
        self.test_ood_topo_actor_seed = test_ood_topo_actor_seed

        self.initial_chronics_id = initial_chronics_id
        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x")\
                     +self.config.get_option("attr_y")


        self.train_dataset = QuasiStaticWheelDataSet("train",
                                              attr_names=attr_names,
                                              config=self.config,
                                              log_path=log_path
                                              )

        if load_data_set:
            self.load()

    def load(self):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets)
        self.is_loaded = True


    def evaluate_simulator(self,
                           dataset: str = "all",
                           augmented_simulator: Union[PhysicalSimulator, AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           **kwargs) -> dict:
        return 0

if __name__ == '__main__':
    print("toto")