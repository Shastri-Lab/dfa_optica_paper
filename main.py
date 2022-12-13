"""This module simulates photonic training of a feedforward neural network using the DFA algorithm 
on the MNIST dataset. Simulation parameters are specified by command line arguments.
This code was based on https://github.com/pytorch/examples/tree/main/mnist"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from dfa import DFALayer, DFAOutput


class OpticalNN(nn.Module):
    def __init__(self, hidden_layers, error_mean, error_std):
        super().__init__()
        layers = [784, *hidden_layers, 10]
        self.fcs = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(hidden_layers) + 1)]
        )
        self.dfa_layers = nn.ModuleList([DFALayer() for _ in range(len(hidden_layers))])
        self.dfa_output = DFAOutput(self.dfa_layers, error_mean, error_std)

    def forward(self, x):
        x = x.reshape(-1, 784)
        for i, dfa_layer in enumerate(self.dfa_layers):
            x = dfa_layer(torch.relu(self.fcs[i](x)))
        x = self.dfa_output(self.fcs[-1](x))
        return x


def train(args, model, device, train_loader, optimizer):
    model.train()
    for (data, target) in tqdm(train_loader, disable=args.no_progressbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if args.dry_run:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Photonic DFA Training")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[800, 800],
        help="Size of the hidden layers",
    )
    parser.add_argument(
        "--error-mean", type=float, default=0, help="Mean error of each MAC operation"
    )
    parser.add_argument(
        "--error-std",
        type=float,
        default=0,
        help="Standard deviation of the error of each MAC operation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help="Input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 1.0)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument(
        "--gamma", type=float, default=1, help="Learning rate step gamma (default: 1)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="Quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For saving the current model"
    )
    parser.add_argument(
        "--no-progressbar", action="store_true", default=False, help="Don't display progress bar"
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_set, test_set = torch.utils.data.random_split(
        datasets.MNIST("./data", train=False, transform=transform), [5000, 5000]
    )
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    model = OpticalNN(args.hidden_layers, args.error_mean, args.error_std).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_acc = 0
    best_model = OpticalNN(args.hidden_layers, args.error_mean, args.error_std).to(device)
    best_model.load_state_dict(model.state_dict())

    print(args)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer)
        val_acc = test(model, device, val_loader)
        print(f"Epoch {epoch} -  Validation accuracy {val_acc}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model.load_state_dict(model.state_dict())
        scheduler.step()

    test_acc = test(best_model, device, test_loader)
    print(f"Test Accuracy: {test_acc}")

    if args.save_model:
        torch.save(best_model.state_dict(), "Optical_NN.pt")


if __name__ == "__main__":
    main()
