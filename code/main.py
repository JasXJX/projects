import util
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# devc = "cuda" if torch.cuda.is_available() else "cpu"
devc = "cpu"
torch.device(devc)
torch.manual_seed(3)

learning_rate = 0.004
epoch_total = 500


def train(model: torch.nn.Module,
          features: torch.Tensor,
          labels: torch.Tensor,
          lr: int = learning_rate,
          ep_n: int = 500
          ):
    model.train()
    train_set = TensorDataset(features, labels)
    loader = DataLoader(train_set, batch_size=5, shuffle=False)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_vals = []
    for e in tqdm(range(ep_n), desc="Training..."):
        loss_val = 0
        for feat, lbl in loader:
            optimizer.zero_grad()
            pred = model(feat)
            loss = loss_func(pred, lbl)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        loss_vals.append(loss_val)
    train_acc = test(model, features, labels, "Calculating Training Acc...")
    print(f"Training Accuracy: {train_acc}")
    plt.plot([i for i in range(ep_n)], loss_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def test(model: torch.nn.Module,
         features: torch.Tensor,
         labels: torch.Tensor,
         message: str
         ) -> float:
    _, pred_max = torch.max(model(features), dim=1)
    # num_correct = torch.sum(pred_max == labels)
    num_correct = 0
    for i in tqdm(range(len(labels)),
                  desc=message
                  ):
        if pred_max[i] == labels[i]:
            num_correct += 1
    return num_correct / len(labels)


if __name__ == "__main__":
    source = util.data_loader("data/train.csv")
    train_x, train_y, valid_x, valid_y = util.spl_and_norm(source)
    model = util.TitanicPred()
    model.cuda() if devc == "cuda" else model.cpu()
    train(model, train_x, train_y)
    valid_acc = test(model, valid_x, valid_y, "Calculating Valid Acc...")
    print(f"Validation Accuracy: {valid_acc}")
