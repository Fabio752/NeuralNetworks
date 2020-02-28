import numpy as np
import pickle
import torch
import sys
import time
import random
import sklearn
import keras
import torch.nn as nn
import pandas
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Activation,Embedding,Flatten,LeakyReLU,BatchNormalization
from keras.activations import relu,sigmoid
from numpy import savetxt

debug = False

def get_accuracy(y_out, y_target, test=False):
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    test_zeros = 0
    test_ones = 0
    y_pred = y_out >= 0.5       # a Tensor of 0s and 1s
    num_correct = torch.sum(y_target==y_pred.float())  # a Tensor
    acc = (num_correct.item() * 100.0 / len(y_target))  # scalar
    if test:
        for i in range(len(y_out)):
            #print("y_target: ", y_target[i], "y_out: ", y_out[i], "y_pred", y_pred[i].float(), "Classification: ", y_target[i]==y_pred[i].float())
            if (y_target[i] == 1.0):
                test_ones += 1
                if (y_pred[i] == 0.0):
                    false_negative += 1
                else:
                    true_positive += 1
            elif (y_target[i] == 0.0):
                test_zeros += 1
                if (y_pred[i] == 1.0):
                    false_positive += 1
                else:
                    true_negative += 1
        return acc, false_negative, true_negative, false_positive, true_positive, test_ones, test_zeros
    else:
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
    def __init__(self, **args):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        print("hello claim classifier here")
        # super(ClaimClassifier, self).__init__()
        self.retrain = False
        self.count_zeros = 0
        self.count_ones = 0
        self.one_indexes = []
        self.zero_indexes = []
        self.loss = 0
        self.learning_rate = 0.001
        self.batch_size = 50
        self.n_epochs = 200
        self.train = None
        self.val = None
        self.test = None
        self.model = nn.Sequential(
            nn.Linear(9,4),
            nn.ReLU(),
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
        )
        for arg, value in args.items():
            setattr(self, arg, value)

    def change_model(self, n_H, act, dr=0.0):
        self.model = nn.Sequential(
            nn.Linear(9,n_H),
            nn.Dropout(dr),
            act,
            nn.Linear(n_H,n_H),
            nn.Dropout(dr),
            act,
            nn.Linear(n_H,n_H),
            nn.Dropout(dr),
            act,
            nn.Linear(n_H,1),
            nn.Sigmoid(),
        )

    def get_params(self, deep=True):
        return {
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "model": self.model
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            print(parameter, value)
            setattr(self, parameter, value)
        return self

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

        self.raw_data = X_raw
        self.n_cols = np.size(self.raw_data, 1)
        self.n_rows = np.size(self.raw_data, 0)
        #Create a temporary copy in order to store only the data the model will be training on
        max = np.max(self.raw_data, axis=0)
        min = np.min(self.raw_data, axis=0)
        #Normalize relevant columns
        self.raw_data = (self.raw_data - min) / (max - min)

        return self.raw_data

    def upsample_ones(self, X, y, w=0):
        ''' Function to upsample the instances of ones (minority class) in the dataset '''
        #Prevent division by zero
        if (self.count_ones == 0):
            return X,y #returning both unaltered
        diff = round(self.count_zeros/self.count_ones)
        original_length = len(X)
        for i in range(self.count_ones):
            for j in range(diff):
                row = X[self.one_indexes[i], :]
                X = np.vstack([X, row])
                y = np.vstack([y, 1.0])
                self.count_ones += 1
                if(self.count_ones >= self.count_zeros):
                    return (X,y)
        return (X, y)

    def downsample_zeros(self, X, y, w):
        original_length = len(X)

        for i in range(len(self.zero_indexes)-1, round(len(self.zero_indexes)*w), -1):
            X = np.delete(X, self.zero_indexes[i], 0)
            y = np.delete(y, self.zero_indexes[i], 0)
            self.count_zeros -= 1
            if (self.count_ones >= self.count_zeros):
                return (X,y)
        return (X,y)


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
        one_indexes = []
        zero_indexes = []

        #Preprocessing the data
        if y_raw is None:
            y = X_raw[:, X_raw.shape[1]-1:]
            X_raw = X_raw[:, :X_raw.shape[1]-2]
        else:
            y = y_raw
            X = X_raw

        X = self._preprocessor(X_raw)

        self.count_zeros = 0
        self.count_ones = 0
        print("Length of y: ", len(y))
        for i in range(len(y)):
            if y[i] == 0.0:
                self.count_zeros += 1
                self.zero_indexes.append(i)
            elif y[i] == 1.0:
                self.count_ones += 1
                self.one_indexes.append(i)

        if debug==True:
            print("Original Zeros Counted: ", self.count_zeros)
            print("Original Ones Counted: ", self.count_ones)

        X, y = self.upsample_ones(X_raw, y)

        self.count_zeros = 0
        self.count_ones = 0
        for i in range(len(y)):
            if y[i] == 0.0:
                self.count_zeros += 1
            elif y[i] == 1.0:
                self.count_ones += 1
                self.one_indexes.append(i)

        if debug:
            print("Down/Upsampled Zeros Counted: ", self.count_zeros)
            print("Down/Upsampled Ones Counted: ", self.count_ones)
            print("Length of X: ", len(X[:, 0]))
            print("Length of y: ", len(y[:, 0]))
        if self.count_ones != 0:
            diff = round(self.count_zeros/self.count_ones)

        #Prepare data to be accepted by Pytorch
        ds = PrepareData(X=X, y=y)
        ds = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        cost_func = nn.BCELoss()

        #Training model
        for epoch in range(self.n_epochs):
            count_zeros = 0
            count_ones = 0
            accuracies = []
            zero_accuracies = []
            losses = []

            for ix, (_x, _y) in enumerate(ds):
                #=========make input differentiable=======================
                _x = Variable(_x).float()
                _y = Variable(_y).float()

                #========forward pass=====================================
                yhat = self.model(_x).float()
                loss = cost_func(yhat, _y)
                accuracy = get_accuracy(yhat, _y)
                zero_accuracy = get_accuracy_zero_tensor(yhat, _y)
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

            if epoch % 1 == 0:
                self.loss = np.average(losses)
                if debug:
                    print("[{}/{}], loss: {} acc: {}".format(epoch,
                    self.n_epochs, np.average(losses), np.average(accuracies)))
                    print("[{}/{}],  zero acc: {}".format(epoch,
                    self.n_epochs, np.average(zero_accuracies)))
                    print("Ones predicted: ", count_ones)
                    print("Zeros predicted: ", count_zeros)
                else:
                    pass
        return self
        # print("Finished Training")


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
        #Taking care of Datarame input
        if isinstance(X_raw, pandas.DataFrame):
            X_raw = X_raw.values

        X_clean = self._preprocessor(X_raw)
        if debug:
            print("Xclean data type: ", type(X_clean))
            print("Data type of the return of preprocessor", type(self._preprocessor(X_raw)))
            print("self.val type: ", type(self.val))
        temp1 = torch.from_numpy(X_clean)
        temp2 = Variable(temp1)
        X_clean = temp2.float()
        #X_clean_tensor = Variable(torch.from_numpy(X_clean)).float()
        # print("Data type on the next line: ", type(X_clean))
        print("==========================================")
        print(X_clean)
        print(X_clean.shape)
        print(self.model)
        print("==========================================")
        out = self.model(X_clean).float()
        out = out.detach().numpy()
        # print(out)
        # print("Out at the last line", type(out))
        return out

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        X_val = self.val[:, :self.val.shape[1]-2]
        y_val = self.val[:, self.val.shape[1]-1:]
        y_val = Variable(torch.from_numpy(y_val)).float()
        out = self.predict(X_val)
        acc, fn, tn, fp, tp, test_ones, test_zeros = get_accuracy(torch.from_numpy(out), y_val, True)
        print("------------------------------------------------------------------")
        print("Misclassified ones: ", fn, "/", test_ones)
        print("------------------------------------------------------------------")
        print("Misclassified zeros: ", fp, "/", test_zeros)
        print("------------------------------------------------------------------")
        print("Accuracy: ", acc)
        print("------------------------------------------------------------------")
        precision = 0
        recall = 0
        f1 = 0
        if (tp+fp) != 0:
            precision = tp/(tp+fp)
        if  (tp+fn) != 0:
            recall = tp/(tp+fn)
        print("Precision -> Pr(positive example|example classified as positive): ", precision)
        print("Recall -> Pr(correctly classified|positive example): ", recall)
        print("------------------------------------------------------------------")
        if (precision+recall) != 0:
            f1 = 2*(precision*recall)/(precision + recall)
        print("F1 Score: ", f1)
        print("------------------------------------------------------------------")
        print("Confusion Matrix")
        print("TP: ", tp, " FN: ", fn)
        print("FP: ", fp, " TN: ", tn)
        print("------------------------------------------------------------------")
        auc_score = sklearn.metrics.roc_auc_score(y_val, out)
        print("AUC-ROC Score: ", auc_score)
        print("------------------------------------------------------------------")
        print("Loss: ", self.loss)
        print("##################################################################")

        return acc, precision, recall, f1, auc_score

    def score(self, X):
        X_val = X[:, :X.shape[1]-2]
        y_val = X[:, X.shape[1]-1:]
        y_val = Variable(torch.from_numpy(y_val)).float()
        out = self.predict(X_val)
        auc_score = 0
        try:
            auc_score = sklearn.metrics.roc_auc_score(y_val, out)
        except:
            auc_score = 0
        print("Score: " + str(auc_score))
        return auc_score

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
def ClaimClassifierHyperParameterSearch(cc, X_train):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    '''
    Hyperparameters to optmize:
    1. Number of neurons and neuron layers
    2. Learning rate
    3. Number of epochs
    4. Input type
    5. Leaky Relu parameters
    6. Batch Size
    7. Batch Normalization
    8. Dropout rate
    '''    
    neurons =[4,6,8,10]
    hidden_layers = [1,2]
    epochs = [25,50,75,100]
    batch_sizes = [10,25,50,100,250,500,1000,2000]
    learning_rates = [0.1,0.01,0.001]
    batch_normalization = [False, True]
    dropout_rates = [0.0,0.1,0.25,0.35,0.5,0.6, 0.75]
    leak_parameters = [0.0, 0.1, 0.35, 0.5, 0.01, 0.05, 0.001]
    best_model = []
    best_auc = 0

    #Only adjusting some of the parameters
    for neuron in neurons:
        for dropout in dropout_rates:
            for leak in leak_parameters:
                print("Number of neurons: ", neuron)
                print("Dropot rate: ", dropout)
                print("Leak rate: ", leak)
                cc.change_model(neuron, nn.LeakyReLU(leak), dropout)
                cc.n_epochs = 25
                cc.fit(X_train, None)
                acc, precision, recall, f1, auc_score = cc.evaluate_architecture()
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_model = []
                    best_model.append(neuron)
                    best_model.append(dropout)
                    best_model.append(leak)
                print("Best model: ", best_model)

    return  # Return the chosen hyper parameters
# path_to_train = "part2_train_.csv"
# path_to_val = "part2_validation.csv"
# path_to_test = "part2_test.csv"
# cc = ClaimClassifier()
# #Extracting from csv
# train_raw = np.genfromtxt(path_to_train, delimiter=',')[1:, :]
# val_raw = np.genfromtxt(path_to_val, delimiter=',')[1:, :]
# #Preprocessing the data
# cc.val = val_raw

# cc.fit(train_raw)
# cc.evaluate_architecture()
# cc.save_model()

class HyperParamSearcher():
    def __init__(self, param_grid, train_data):
        self.cc = GridSearchCV(ClaimClassifier(), param_grid, cv=3)
        self.train_data = train_data

    def begin(self):
        self.cc.fit(self.train_data)
        self.end()

    def end(self):
        print(self.cc.best_params_)
        print(self.cc.best_score_)
#path_to_train = "part2_train_.csv"
'''path_to_train = "upsampled_dataset.csv"
path_to_val = "part2_validation.csv"
path_to_test = "part2_test.csv"
cc = ClaimClassifier()
# #Extracting from csv
train_raw = np.genfromtxt(path_to_train, delimiter=',')[1:, :]
8519
#cc.fit(train_raw)
#cc.evaluate_architecture()
# cc.save_model()
tuned_parameters = [{ 'n_epochs': [50], 'model': [nn.Sequential(nn.Linear(9,4),nn.ReLU(),nn.Linear(4,4),nn.ReLU(),nn.Linear(4,1),nn.Sigmoid(),), 
nn.Sequential(nn.Linear(9,5),nn.ReLU(),nn.Linear(5,5),nn.ReLU(),nn.Linear(5,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,6),nn.ReLU(),nn.Linear(6,6),nn.ReLU(),nn.Linear(6,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,7),nn.ReLU(),nn.Linear(7,7),nn.ReLU(),nn.Linear(7,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),nn.Linear(8,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,9),nn.ReLU(),nn.Linear(9,9),nn.ReLU(),nn.Linear(9,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,10),nn.ReLU(),nn.Linear(10,10),nn.ReLU(),nn.Linear(10,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,25),nn.ReLU(),nn.Linear(25,25),nn.ReLU(),nn.Linear(25,1),nn.Sigmoid(),),
nn.Sequential(nn.Linear(9,50),nn.ReLU(),nn.Linear(50,50),nn.ReLU(),nn.Linear(50,1),nn.Sigmoid(),), 
nn.Sequential(nn.Linear(9,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,1),nn.Sigmoid(),)]}]
searcher = HyperParamSearcher(tuned_parameters, train_raw)
searcher.begin()'''
