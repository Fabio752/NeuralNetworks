import numpy as np
import pickle
import torch
import sys
import time
import random
import sklearn
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score



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
    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        super(ClaimClassifier, self).__init__()
        self.n_H = 7
        self.activation = nn.ReLU()
        self.dropout_rate = 0.0
        self.count_zeros = 0
        self.count_ones = 0
        self.one_indexes = []
        self.zero_indexes = []
        self.train = None
        self.val = None
        self.test = None
        self.model = nn.Sequential(
            nn.Linear(9,self.n_H),
            #nn.Dropout(self.dropout_rate),
            self.activation,
            nn.Linear(self.n_H,self.n_H),
            #nn.Dropout(self.dropout_rate),
            self.activation,
            nn.Linear(self.n_H,1),
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

        #TODO Check if data normalisation is correct in the new way (also passing CSV and that nonesense)

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
        #Shuffle to remove bias in the dataset
        np.random.shuffle(self.raw_data)
        return self.raw_data 

    def upsample_ones(self, X, y, w):
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
                if(len(y) == round((w)*original_length)):
                    return (X,y)
        return (X, y)

    def downsample_zeros(self, X, y, w):
        original_length = len(X)
        for i in range(len(self.zero_indexes)-1, round(len(self.zero_indexes)*w), -1):
            X = np.delete(X, self.zero_indexes[i], 0)
            y = np.delete(y, self.zero_indexes[i], 0)
            
        return (X,y)
                    

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
        zero_indexes = []

        X = X_raw[:, :self.n_cols - 2]
        y = X_raw[:, self.n_cols-1:]
        
        
        losses = []
        accuracies = []
        zero_accuracies = []

        for i in range(len(y)):
            if y[i] == 0.0:
                self.count_zeros += 1
                self.zero_indexes.append(i)
            elif y[i] == 1.0:
                self.count_ones += 1
                self.one_indexes.append(i)
        print("Original Zeros Counted: ", self.count_zeros)
        print("Original Ones Counted: ", self.count_ones)

        #Upsampling
        X_ = self.upsample_ones(X, y, 1.5)

        X = X_[0]
        y = X_[1]


        #Downsampling
        X_ = self.downsample_zeros(X, y, 0.65)

        #Shuffle the array in order to remove bias

        X = X_[0]
        y = X_[1]


        self.count_zeros = 0
        self.count_ones = 0
        for i in range(len(y)):
            if y[i] == 0.0:
                self.count_zeros += 1
            elif y[i] == 1.0:
                self.count_ones += 1
                self.one_indexes.append(i)
                
        print("Down/Upsampled Zeros Counted: ", self.count_zeros)
        print("Down/Upsampled Ones Counted: ", self.count_ones)

        print("Length of X: ", len(X[:, 0]))
        print("Length of y: ", len(y[:, 0]))

        #assert(False)


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
            count_zeros = 0
            count_ones = 0
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
        #Run the model on test set
        out = self.model(X_val).float()
        y_val = cc.val[:, cc.val.shape[1]-1:]
        y_val = Variable(torch.from_numpy(y_val)).float()
        out = out.detach()

        acc, fn, tn, fp, tp, test_ones, test_zeros = get_accuracy(out, y_val, True)
        print("------------------------------------------------------------------")
        print("Accuracy: ", acc)
        print("------------------------------------------------------------------")
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print("Precision -> Pr(positive example|example classified as positive): ", precision)
        print("Recall -> Pr(correctly classified|positive example): ", recall)
        print("------------------------------------------------------------------")
        f1 = 2*(precision*recall)/(precision + recall)
        print("F1 Score: ", f1)
        print("------------------------------------------------------------------")
        print("Confusion Matrix")
        print("TP: ", tp, " FN: ", fn)
        print("FP: ", fp, " TN: ", tn)
        print("------------------------------------------------------------------")
        auc_score = sklearn.metrics.roc_auc_score(y_val, out)
        print("AUC-ROC Score: ", auc_score)

        return acc, precision, recall, f1, auc_score


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(cc, X_test, y_test):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    return  # Return the chosen hyper parameters


path_to_train = "part2_train_.csv"
path_to_val = "part2_validation.csv"
path_to_test = "part2_test.csv"
cc = ClaimClassifier()
#data preprocessing
cc.train = cc._preprocessor(path_to_train)

acc = []
cost_func = nn.BCELoss()
#Training the model
cc.fit(cc.train, None, 250)

#Assigning validation set
cc.val = cc._preprocessor(path_to_val)
y_val = cc.val[:, cc.val.shape[1]-1:]
X_val = cc.val[:, :cc.val.shape[1]-2]
#Test the data
X_val = Variable(torch.from_numpy(X_val)).float()
y_val = Variable(torch.from_numpy(y_val)).float()

#Run through the generated model
cc.evaluate_architecture()
#acc = get_accuracy(out, y_val)
#loss = cost_func(out, y_val)

#print("Acc: ", acc)
#print("Loss(BCE): ", loss)




