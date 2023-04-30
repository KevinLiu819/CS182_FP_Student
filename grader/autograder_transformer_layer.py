import torch
import numpy as np

TRANSFORMER_LAYER_FILE = "grader/transformer_layer_test.npz"
TRANSFORMER_LAYER_SUB_FILE = "grader/transformer_layer_test_sub.npz"

def grade_transformer_layer(transformer_layer_class):
    data = np.load(TRANSFORMER_LAYER_FILE)
    seed = data["seed"]
    transformer_layer_in = torch.tensor(data["input"])
    transformer_layer_out = data["output"]

    torch.manual_seed(seed)
    transformer_layer = transformer_layer_class(1024)
    out = transformer_layer(transformer_layer_in)
    out = out.detach().numpy()
    
    assert out.shape == transformer_layer_out.shape
    a, b, _ = out.shape
    for i in range(a):
        for j in range(b):
            assert np.allclose(out[i, j], transformer_layer_out[i, j], atol=1e-2)

def generate_transformer_layer_sub(transformer_layer_class):
    data = np.load(TRANSFORMER_LAYER_SUB_FILE)
    seed = data["seed"]
    transformer_layer_in = torch.tensor(data["input"])

    torch.manual_seed(seed)
    transformer_layer = transformer_layer_class(1024)
    out = transformer_layer(transformer_layer_in)
    out = out.detach().numpy()
    return out