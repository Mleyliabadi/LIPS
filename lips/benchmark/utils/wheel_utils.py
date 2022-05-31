"""
Rolling Wheel general scenario utilities
"""

from lips.config.configmanager import ConfigManager

def get_kwargs_simulator_scenario(config: ConfigManager) -> dict:
    """Return environment parameters for Benchmark1

    Parameters
    ----------
    config : ConfigManager
        ``ConfigManager`` instance comprising the options for a scenario

    Returns
    -------
    dict
        the dictionary of parameters
    """
    env_name = config.get_option("env_name")
    param = Parameters()
    param.init_from_dict(config.get_option("env_params"))
    return {"dataset": env_name,
            "param": param,
            "backend": BkCls()}