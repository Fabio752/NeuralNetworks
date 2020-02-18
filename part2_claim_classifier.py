import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import sys


def get_accuracy(y_out, y_target):
  y_pred = y_out >= 0.5       # a Tensor of 0s and 1s
  num_correct = torch.sum(y_target==y_pred.float())  # a Tensor
  acc = (num_correct.item() * 100.0 / len(y_target))  # scalar
  return acc


class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClaimClassifier():
    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        super(ClaimClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid(),
        )

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """

        self.raw_data = np.genfromtxt(X_raw, delimiter=',')[1:, :]
        self.n_cols = np.size(self.raw_data, 1)
        max = np.max(self.raw_data, axis=0)
        min = np.min(self.raw_data, axis=0)
        return (self.raw_data - min) / (max - min)


    def fit(self, X_raw, y_raw=None):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """


        '''if y_raw is None: <--DEAL WITH THIS!'''

        #data preprocessing
        X_clean = self._preprocessor(X_raw)
        X = X_clean[:, :self.n_cols - 2]
        #y = to_categorical(X_clean[:, self.n_cols-1:],2)
        y = X_clean[:, self.n_cols-1:]
        ds = PrepareData(X=X, y=y)
        ds = DataLoader(ds, batch_size=50, shuffle=True)

        n_epochs = 30
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        cost_func = nn.BCELoss()

        losses = []
        accuracies = []

        for epoch in range(n_epochs):
            for ix, (_x, _y) in enumerate(ds):
                #=========make inpur differentiable=======================
                _x = Variable(_x).float()
                _y = Variable(_y).float()

                #========forward pass=====================================
                yhat = self.model(_x).float()
                loss = cost_func(yhat, _y)
                accuracy = get_accuracy(yhat, _y)
                #acc = torch.eq(yhat.round(), _y).float().mean() # accuracy

                #=======backward pass=====================================
                self.model.zero_grad() # zero the gradients on each pass before the update
                loss.backward() # backpropagate the loss through the model
                optimizer.step() # update the gradients w.r.t the loss

                #print(loss.data)

                losses.append(loss.data)
                accuracies.append(accuracy)

                '''for n,x in self.model.named_modules():
                    if isinstance(x, nn.Linear):
                        print(x.__dict__.keys())

                print(optimizer.__dict__.keys())'''
                #breakpoint()

            if epoch % 1 == 0:
                print("[{}/{}], loss: {} acc: {}".format(epoch,
                n_epochs, np.average(losses), np.average(accuracies)))



    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    return  # Return the chosen hyper parameters


path_to_data = "part2_training_data.csv"

cc = ClaimClassifier()
print(cc._preprocessor(path_to_data))

cc.fit(path_to_data)
