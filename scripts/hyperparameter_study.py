import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net
import optuna
import optuna.visualization as ov
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io
matplotlib.use("TkAgg")  # or any other backend


def objective(trial):
    # Set up the trial's parameters to search
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 1, 10)

    # Training settings
    args = argparse.Namespace(
        batch_size=64,
        test_batch_size=1000,
        epochs=epochs,
        lr=learning_rate,
        gamma=0.7,
        no_cuda=False,
        dry_run=False,
        seed=1,
        log_interval=10,
        save_model=False,
    )
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader, epoch)

    return test_acc


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return 100.0 * correct / len(test_loader.dataset)


if __name__ == "__main__":
    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Visualize the optimization process
    # Generate the figures using Optuna
    fig = ov.plot_optimization_history(study)
    fig2 = ov.plot_slice(study)

    # Convert Plotly figures to static images
    image_bytes = fig.to_image(format="png")
    image_bytes2 = fig2.to_image(format="png")

    # Convert image bytes to PIL Image objects
    image = Image.open(io.BytesIO(image_bytes))
    image2 = Image.open(io.BytesIO(image_bytes2))

    # Display the images using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(image2)
    plt.axis("off")
    plt.show()
