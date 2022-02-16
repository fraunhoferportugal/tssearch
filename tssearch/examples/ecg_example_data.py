import tssearch
import pickle
import numpy as np


def load_ecg_example():

    filename = tssearch.__path__[0] + "/examples/ecg.pickle"
    with open(filename, "rb") as handle:
        data = pickle.load(handle)

    return data
