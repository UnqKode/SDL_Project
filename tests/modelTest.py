from utils.load_global_prototypes import load_global_prototypes
import numpy as np

def model_trained(trained_model):
    assert trained_model.is_trained == True


def test_prototypes():
    prototypes,all_classes = load_global_prototypes()
    assert prototypes["prototypes"] is not None
    assert prototypes["classes"] is not None
    assert prototypes["classes"].shape[1] == 6
    assert np.ndim == 2
    assert prototypes["prototypes"].shape == (128,6)

