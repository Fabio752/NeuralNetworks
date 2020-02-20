import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import sys
import time


def get_accuracy(y_out, y_target):
  y_pred = y_out >= 0.5       # a Tensor of 0s and 1s
  num_correct = torch.sum(y_target==y_pred.float())  # a Tensor
  acc = (num_correct.item() * 100.0 / len(y_target))  # scalar
  return acc

def get_accuracy_zero_tensor(y_out, y_target):
  y_pred = y_out > 1       # a Tensor of 0s 
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
        self.n_rows = np.size(self.raw_data, 0)
        #Create a temporary copy in order to store only the data the model will be training on
        raw_data_temp = self.raw_data[:, : self.n_cols-2]
        max = np.max(raw_data_temp, axis=0)
        min = np.min(raw_data_temp, axis=0)
        #Normalize relevant columns
        raw_data_temp = (raw_data_temp - min) / (max - min)
        self.raw_data[:, : self.n_cols-2] = raw_data_temp
        return self.raw_data 


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

        
        one_indexes = []

        #data preprocessing
        X_clean = self._preprocessor(X_raw)

        X_ = X_clean[:, self.n_cols - 1]
        #TODO Get rid of the below and reimplement properly
        y_test = X_[round(self.n_rows*0.8): ]

        #Count the number of occurences of zeros and ones, using a for loop in order to count indexes as well
        count_zeros = 0
        count_ones = 0
        for i in range(len(X_)):
            if X_[i] == 0.0:
                count_zeros += 1
            else:
                count_ones += 1
                one_indexes.append(i)

        diff = round(count_zeros/count_ones)

        #TODO FIt this into an oversampling function
        
        #Upsample cases of made_claim == 1 in order to get a balanced dataset
        for i in range(count_ones):
            for j in range(round(diff/6)):
                row = X_clean[one_indexes[i], :]
                if j == 0:
                    x_upsample = np.vstack([row, row])
                else:
                    x_upsample = np.vstack([x_upsample, row])
            X_clean = np.vstack([X_clean, x_upsample])

        X_ = X_clean[:, self.n_cols - 1]
        count_zeros = 0
        count_ones = 0
        for i in range(len(X_)):
            if X_[i] == 0.0:
                count_zeros += 1
                one_indexes.append(i)
            else:
                count_ones += 1
        diff = round(count_zeros/count_ones)
        print("Ones: ", count_ones)
        print("Zeros: ", count_zeros)
        print("Diff: ", diff)


        X = X_clean[:, :self.n_cols - 2]
        #Shuffle the array in order to remove bias
        np.random.shuffle(X)
        #TODO comment this out later
        #X_train = X[ :round(self.n_rows*0.8), :]
        #X_test = X[round(self.n_rows*0.8): , :]
        #X = X_train
        y = X_clean[:, self.n_cols-1:]
        ds = PrepareData(X=X, y=y)
        ds = DataLoader(ds, batch_size=50, shuffle=True)

        n_epochs = 30
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        cost_func = nn.BCELoss()

        losses = []
        accuracies = []
        zero_accuracies = []

        #Training model 
        for epoch in range(n_epochs):
            for ix, (_x, _y) in enumerate(ds):
                #=========make inpur differentiable=======================
                _x = Variable(_x).float()
                _y = Variable(_y).float()

                #========forward pass=====================================
                yhat = self.model(_x).float()
                loss = cost_func(yhat, _y)
                accuracy = get_accuracy(yhat, _y)
                zero_accuracy = get_accuracy_zero_tensor(yhat, _y)
                #acc = torch.eq(yhat.round(), _y).float().mean() # accuracy

                #=======backward pass=====================================
                self.model.zero_grad() # zero the gradients on each pass before the update
                loss.backward() # backpropagate the loss through the model
                optimizer.step() # update the gradients w.r.t the loss

                #print(loss.data)

                losses.append(loss.data)
                accuracies.append(accuracy)
                zero_accuracies.append(zero_accuracy)

                '''for n,x in self.model.named_modules():
                    if isinstance(x, nn.Linear):
                        print(x.__dict__.keys())

                print(optimizer.__dict__.keys())'''
                #breakpoint()

            if epoch % 1 == 0:
                print("[{}/{}], loss: {} acc: {}".format(epoch,
                n_epochs, np.average(losses), np.average(accuracies)))
                print("[{}/{}],  zero acc: {}".format(epoch,
                n_epochs, np.average(zero_accuracies)))


        #Check performance on the training set
        yhat = self.model(X_test).float()
        accuracy = get_accuracy(yhat, y_test)
        zero_accuracy = get_accuracy_zero_tensor(yhat, y_test)
        print("acc after testing: {}".format(accuracy))
        print("zero acc after testing: {}".format(zero_accuracy))



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
cc.fit(path_to_data)
