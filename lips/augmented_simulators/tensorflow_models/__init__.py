__all__ = ["TfFullyConnected"]

from lips.augmented_simulators.tensorflow_models.fully_connected import TfFullyConnected

try:
    from lips.augmented_simulators.tensorflow_models.fullyConnectedAS import FullyConnectedAS
    __all__.append("FullyConnectedAS")
except ImportError:
    # tensorflow package is not installed i cannot used this augmented simulator
    pass


try:
    from lips.augmented_simulators.tensorflow_models.leapNetAS import LeapNetAS
    __all__.append("LeapNetAS")
except ImportError:
    # leap_net package is not installed i cannot used this augmented simulator
    pass
