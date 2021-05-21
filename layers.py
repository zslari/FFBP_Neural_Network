from enum import Enum


class LayerType(Enum):
    """ Used to differentiate Input, Hidden, and Output Neurodes """
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2
