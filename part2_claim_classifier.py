import numpy as np
import pickle
import torch
import torch.nn as nn


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

        X_clean = self._preprocessor(X_raw)


        if y_raw is None:
            np.random.shuffle(X_clean)
            split_idx = int(0.8*len(X_clean))

            x_train = torch.tensor(X_clean[:split_idx, :self.n_cols-2], dtype=torch.float32)
            x_val = X_clean[split_idx:, :self.n_cols-2]
            y_raw = X_clean[:, self.n_cols-1:]

            y_train = torch.tensor(y_raw[:split_idx], dtype=torch.float32)
            y_val = y_raw[split_idx:]

        else: #not sure about this bit
            split_idx = int(0.8*len(X_clean))

            x_train = X_clean[:split_idx, :self.n_cols-2]
            x_val = X_clean[split_idx:, :self.n_cols-2]

            y_train = y_raw[:split_idx]
            y_val = y_raw[split_idx:]



        # Defines number of epochs
        n_epochs = 5
        # Optimizers require the parameters to optimize and a learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.003)

        for epoch in range(n_epochs):
            # Training pass
            optimizer.zero_grad()
            # Computes our model's predicted output
            prediction = self.model(x_train)
            print(prediction)
            pred_y = torch.tensor((prediction >= 0.5), dtype=torch.float32, requires_grad=True)
            print(pred_y)

            criterion = nn.BCELoss()
            loss = criterion(pred_y, y_train.float())
            running_loss = 0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Training loss: {running_loss}")


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
