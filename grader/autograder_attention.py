import torch
import numpy as np

ATTENTION_FILE = "grader/attention_test.npz"
ATTENTION_SUB_FILE = "grader/attention_test_sub.npz"

def grade_attention(attention_class):
    data = np.load(ATTENTION_FILE)
    seed = data["seed"]
    attention_in = torch.tensor(data["input"])
    attention_out = data["output"]

    torch.manual_seed(seed)
    attention_layer = attention_class(1024)
    attention = attention_layer(attention_in)
    out = attention.detach().numpy()
    
    assert out.shape == attention_out.shape
    a, b, _ = out.shape
    for i in range(a):
        for j in range(b):
            assert np.allclose(out[i, j], attention_out[i, j], atol=1e-2)

def generate_attention_sub(attention_class):
    data = np.load(ATTENTION_SUB_FILE)
    seed = data["seed"]
    attention_in = torch.tensor(data["input"])

    torch.manual_seed(seed)
    attention_layer = attention_class(1024)
    attention = attention_layer(attention_in)
    out = attention.detach().numpy()
    return out