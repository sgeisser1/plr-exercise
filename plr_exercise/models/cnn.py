import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A class to represent a simple CNN model for classifying MNIST digits.

    Methods:
    --------
        forward(x) :
            Forward pass through the CNN model.

    """

    def __init__(self):
        """
        Initialize the CNN model.

        Attributes
        ----------
            None
        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Attributes
        ----------
            x : torch.Tensor
                input data

        Returns
        -------
            torch.Tensor
                output of the model
        """

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
