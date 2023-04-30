import torch
from torch.utils.data import DataLoader

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {"tokens": torch.randint(0, 100, (20,)), "labels": torch.randint(0, 100, (20,))}

def test_train_gpt2(GPT2, train_gpt):
    vocab_size = 100
    d_model = 64
    n_layers = 6
    n_heads = 4
    max_len = 20
    dropout = 0.1
    device = torch.device("cpu")

    model = GPT2(vocab_size, d_model, n_layers, n_heads, max_len, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataloader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    val_dataloader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    epochs = 1
    
    try:
        train_gpt(model, optimizer, train_dataloader, val_dataloader, epochs, device)
        print("Training function implementation passed the test.")
    except Exception as e:
        raise AssertionError("Error during training:", e)

if __name__ == "__main__":
    test_train_gpt2()
