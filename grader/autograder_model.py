import sys
sys.path.append("..")

import torch
from nanogpt import GPT2

def test_gpt2_model():
    vocab_size = 100
    d_model = 64
    n_layers = 6
    n_heads = 4
    dropout = 0.1
    
    model = GPT2(vocab_size, d_model, n_layers, n_heads, dropout)
    
    assert isinstance(model, torch.nn.Module), "GPT2 should be a subclass of torch.nn.Module."
    
    tokens = torch.randint(0, vocab_size, (2, 20))
    output = model(tokens)
    
    assert output.shape == (2, 20, vocab_size), f"Expected output shape (2, 20, {vocab_size}), got {output.shape} instead."

    print("GPT2 model implementation passed the test.")

if __name__ == "__main__":
    test_gpt2_model()
