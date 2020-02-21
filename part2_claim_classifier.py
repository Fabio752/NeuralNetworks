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
            nn.Linear(5,50),
            nn.ReLU(),
            nn.Linear(50,250),
            nn.ReLU(),
            nn.Linear(250,250),
            nn.ReLU(),
            nn.Linear(250,250),
            nn.ReLU(),
            nn.Linear(250,250),
            nn.ReLU(),
            nn.Linear(250,5),
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

    def upsample_ones(self, X, y, count_ones, count_zeros, one_indexes):
        ''' Function to upsample the instances of ones (minority class) in the dataset '''
        #Prevent division by zero
        if (count_ones == 0):
            return X,y #returning both unaltered
        diff = round(count_zeros/count_ones)
        original_length = len(X)
        for i in range(count_ones):
            for j in range(diff):
                row = X[one_indexes[i], :]
                X = np.vstack([X, row])
                y = np.vstack([y, 1.0])
                if(len(y) == (6.5/4)*original_length):
                    return (X,y)
        return (X, y)



    def fit(self, X_raw, y_raw=None, n_epochs=6):
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

        #TODO Comment this out
        one_indexes = []

        X = X_raw[:, :self.n_cols - 2]
        y = X_raw[:, self.n_cols-1:]
        
        
        losses = []
        accuracies = []
        zero_accuracies = []

        count_zeros = 0
        count_ones = 0
        for i in range(len(y)):
            if y[i] == 0.0:
                count_zeros += 1
            elif y[i] == 1.0:
                count_ones += 1
                one_indexes.append(i)
        print("Original Zeros Counted: ", count_zeros)
        print("Original Ones Counted: ", count_ones)

        #Upsampling
        X_ = self.upsample_ones(X, y, count_ones, count_zeros, one_indexes)

        #Shuffle the array in order to remove bias
        #np.random.shuffle(X)

        X = X_[0]
        y = X_[1]

        count_zeros = 0
        count_ones = 0
        for i in range(len(y)):
            if y[i] == 0.0:
                count_zeros += 1
            elif y[i] == 1.0:
                count_ones += 1
                one_indexes.append(i)
                
        print("Upsampled Zeros Counted: ", count_zeros)
        print("Upsampled Ones Counted: ", count_ones)

        print("Length of X: ", len(X[:, 0]))
        print("Length of y: ", len(y[:, 0]))


        #Shuffle the array in order to remove bias
        # TODO Join the two and then shuffle them       
        #np.random.shuffle(X)

        #Prepare data to be accepted by Pytorch
        ds = PrepareData(X=X, y=y)
        ds = DataLoader(ds, batch_size=50, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        cost_func = nn.BCELoss()

        #Training model 
        for epoch in range(n_epochs):
            for ix, (_x, _y) in enumerate(ds):
                #=========make input differentiable=======================
                _x = Variable(_x).float()
                _y = Variable(_y).float()

                #========forward pass=====================================
                yhat = self.model(_x).float()
                loss = cost_func(yhat, _y)
                accuracy = get_accuracy(yhat, _y)
                zero_accuracy = get_accuracy_zero_tensor(yhat, _y)
                #acc = torch.eq(yhat.round(), _y).float().mean() # accuracy

                #=========iterate to find predicted zeros and ones========
                for i in range(len(yhat)):
                    if yhat[i] < 0.5:
                        count_zeros += 1
                    else:
                        count_ones += 1
                #=======backward pass=====================================
                self.model.zero_grad() # zero the gradients on each pass before the update
                loss.backward() # backpropagate the loss through the model
                optimizer.step() # update the gradients w.r.t the loss

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
                print("Ones predicted: ", count_ones)
                print("Zeros predicted: ", count_zeros)
                count_zeros = 0
                count_ones = 0

        print("Finished Training")


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
#data preprocessing
X_clean = cc._preprocessor(path_to_data)

np.random.shuffle(X_clean)
X_train = X_clean[:round(cc.n_rows*0.8) , :]
X_test = X_clean[round(cc.n_rows*0.8):, : X_clean.shape[1]-2]
y_test = X_clean[round(cc.n_rows*0.8):, X_clean.shape[1]-1:]

cc.fit(X_train, None, 150)

#Test the data
X_test = Variable(torch.from_numpy(X_test))
y_test = Variable(torch.from_numpy(y_test))
#Trasnposing input in order for it to be accepted
#X_test = X_test.T
out = cc.model(X_test.float())
accuracy = get_accuracy(out, y_test)
print("Accuracy: ", accuracy)
