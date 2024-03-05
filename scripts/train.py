from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net
import wandb
import optuna
import logging
import sys

wandb.login()


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Train the model for one epoch.

    Attributes
    ----------
        args : argparse.Namespace
            command-line arguments
        model : torch.nn.Module
            model to train
        device : torch.device
            device to use for training
        train_loader : torch.utils.data.DataLoader
            training data
        optimizer : torch.optim.Optimizer
            optimizer to use
        epoch : int
            current epoch number
    """

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss})
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


def test(model, device, test_loader):
    """
    Test the model on the test set.

    Attributes
    ----------
        model : torch.nn.Module
            model to test
        device : torch.device
            device to use for testing
        test_loader : torch.utils.data.DataLoader
            test data

    Returns
    -------
        float
            test accuracy of the model
    """

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
    wandb.log({"test_loss": test_loss, "test_accuracy": 100.0 * correct / len(test_loader.dataset)})

    return 100.0 * correct / len(test_loader.dataset)


def objective(trial, args, model, device, train_loader, test_loader):
    """
    Objective function for Optuna hyperparameter optimization study of the learning rate and epochs.

    Attributes
    ----------
        trial : optuna.Trial
            a single optimization step
        args : argparse.Namespace
            command-line arguments
        model : torch.nn.Module
            model to train
        device : torch.device
            device to use for training
        train_loader : torch.utils.data.DataLoader
            training data
        test_loader : torch.utils.data.DataLoader
            test data

    Returns
    -------
        float
            test accuracy of the model
    """
    # Set up the trial's parameters to search
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 0.1)
    epochs = trial.suggest_int("epochs", 1, 15)

    args.lr = learning_rate
    args.epochs = epochs

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run = wandb.init(
        # Set the project where this run will be logged
        project="plr_exercise",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
        },
        name=f"{args.optuna_study_name}-{trial.number}",
    )

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        trial.report(test_acc, epoch)

    wandb.finish()

    return test_acc


def main():
    """
    Main function to train the model

    Arguments
    ---------
        args : argparse.Namespace
            command-line arguments

    Command line arguments
    ----------------------
        --batch-size : int
            input batch size for training (default: 64)
        --test-batch-size : int
            input batch size for testing (default: 1000)
        --epochs : int
            number of epochs to train (default: 7)
        --lr : float
            learning rate (default: 0.00015)
        --gamma : float
            Learning rate step gamma (default: 0.7)
        --no-cuda : bool
            disables CUDA training (default: False)
        --dry-run : bool
            quickly check a single pass (default: False)
        --seed : int
            random seed (default: 1)
        --log-interval : int
            how many batches to wait before logging training status (default: 10)
        --save-model : bool
            For Saving the current Model (default: False)
        --optuna_optimization : bool
            enable Optuna hyperparameter optimization study (default: False)
        --optuna_study_name : str
            name of the Optuna study (default: optuna-study)
        --num_trials : int
            number of trials for Optuna study (default: 15)
    """
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=7, metavar="N", help="number of epochs to train (default: 7)")
    parser.add_argument("--lr", type=float, default=0.00015, metavar="LR", help="learning rate (default: 0.00015)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument(
        "--optuna_optimization",
        action="store_true",
        default=False,
        help="enable Optuna hyperparameter optimization study",
    )
    parser.add_argument("--optuna_study_name", type=str, default="optuna-study", help="name of the Optuna study")
    parser.add_argument("--num_trials", type=int, default=15, help="number of trials for Optuna study")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

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

    if args.optuna_optimization:  # Optimize hyperparameters using Optuna
        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = args.optuna_study_name
        storage_name = "sqlite:///{}.db".format(study_name)

        study = optuna.create_study(
            direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True
        )
        study.optimize(
            lambda trial: objective(trial, args, model, device, train_loader, test_loader),
            n_trials=args.num_trials,
        )

        # Print the best hyperparameters
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        run = wandb.init(
            # Set the project where this run will be logged
            project="plr_exercise",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
            },
        )
        code_artifact = wandb.Artifact(name="code_snapshot", type="code")
        code_artifact.add_file("scripts/train.py")
        code_artifact.add_file("plr_exercise/models/cnn.py")
        run.log_artifact(code_artifact)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(model, device, test_loader)
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
