import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
from nanogpt import GPT2, train_gpt

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"tokens": torch.randint(0, 100, (20,)), "labels": torch.randint(0, 100, (20,))}

def test_train_gpt2():
    vocab_size = 100
    d_model = 64
    n_layers = 6
    n_heads = 4
    dropout = 0.1
    device = torch.device("cpu")

    model = GPT2(vocab_size, d_model, n_layers, n_heads, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    epochs = 1
    
    try:
        train_gpt(model, optimizer, criterion, dataloader, epochs, device)
        print("Training function implementation passed the test.")
    except Exception as e:
        raise AssertionError("Error during training:", e)

if __name__ == "__main__":
    test_train_gpt2()
