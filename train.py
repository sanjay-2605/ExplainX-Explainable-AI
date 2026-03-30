import torch
import torch.optim as optim
import torch.nn as nn
from models.model import SimpleNN
from utils.data_loader import load_data

def train():
    X_train, X_test, y_train, y_test = load_data()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    model = SimpleNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
