"""
Wheel general scenario utilities
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
    return config.get_option("env_params")