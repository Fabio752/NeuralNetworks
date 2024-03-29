from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from part2_claim_classifier import ClaimClassifier
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import torch
import time

picklefilename = 'part3_pricing_model.pickle'

def get_accuracy(y_out, y_target, test=False):
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0
    test_zeros = 0
    test_ones = 0
    y_pred = y_out >= 0.5       # a Tensor of 0s and 1s
    num_correct = torch.sum(y_target.float()==y_pred.float())  # a Tensor
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



def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, **args):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        # self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.model = ClaimClassifier(**args)  
        self.cols_to_drop = ["id_policy", "pol_duration", "pol_pay_freq", \
        "pol_payd", "pol_usage", "pol_insee_code", "drv_drv2", \
        "drv_age1", "drv_age1","drv_age2", "drv_sex1", "drv_sex2", "drv_age_lic1", \
        "vh_cyl", "vh_make", "vh_model", "drv_age_lic2", \
        "town_mean_altitude", "town_surface_area", "population", "commune_code", \
        "regional_department_code", "canton_code", "city_district_code", "vh_type"]

   
    # the test set might not have all the required columns
    # so, we hardcode the required cols after 1-hot encoding
    # we also make sure the columns are in the correct order
    def add_missing_dummy_cols(self, d):
        # we drop some extra columns as well that have really bad corr
        columns = [ 
                # "pol_bonus",
                "pol_sit_duration",      
                "vh_age",                
                "vh_din",                
                "vh_sale_begin",         
                "vh_sale_end",           
                "vh_speed",              
                "vh_value",              
                "vh_weight",            
                "pol_coverage_Maxi",     
                # "pol_coverage_Median1",  
                # "pol_coverage_Median2",  
                "pol_coverage_Mini",     
                "vh_fuel_Diesel",        
                "vh_fuel_Gasoline",      
                # "vh_fuel_Hybrid"
                ]   
        missing_cols = set(columns) - set(d.columns)
        for c in missing_cols: d[c] = 0
        d = d[columns]

        # new columns by feature engineering -> speed/power ratio
        d["SPR"] = d["vh_speed"] / d["vh_din"]

        return d


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
.values
        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        #Converting the input ndarray into a pandas dataframe

        X_dropped = X_raw.drop(self.cols_to_drop, axis=1)

        X_unscaled_pd = pd.get_dummies(X_dropped, columns=['pol_coverage', 'vh_fuel'])

        X_unscaled_cols = self.add_missing_dummy_cols(X_unscaled_pd)

        X_unscaled = X_unscaled_cols.values



        return X_unscaled

    def fit(self, X_raw, y_raw, claims_raw):
        """
        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        #Dropping last two columns and then preprocesing the data
        X_raw = X_raw.drop(["made_claim", "claim_amount"], axis=1)
        X_clean = self._preprocessor(X_raw)
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        self.model = self.model.fit(X_clean, y_raw)
        return self.model

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # print("Dataframe passed to the predict function:")
        # print(X_raw.info())

        X_clean = self._preprocessor(X_raw)
        probabilities = self.model.predict(X_clean)

        return probabilities.flatten()

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        # print("Xraw in predict premium function:")
        # print(X_raw.info())

        return (self.predict_claim_probability(X_raw) * self.y_mean).flatten()

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open(picklefilename, 'wb') as target:
            pickle.dump(self, target)

    def evaluate_architecture(self, X_val, y_val):
        #Dropping last two columns and then preprocesing the data
        X_val = X_val.drop(["made_claim", "claim_amount"], axis=1)
        preds = self.predict_claim_probability(X_val)
        print("Predictions:")
        print(preds)
        y_val = torch.from_numpy(y_val.values)
        _, fn, tn, fp, tp, test_ones, test_zeros = get_accuracy(torch.from_numpy(preds), y_val, True)
        print("------------------------------------------------------------------")
        print("Misclassified ones: ", fn, "/", test_ones)
        print("------------------------------------------------------------------")
        print("Misclassified zeros: ", fp, "/", test_zeros)
        print("------------------------------------------------------------------")
        acc = (tp+tn)/(tn+tp+fn+fp)
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
        auc_score = roc_auc_score(y_val, preds)
        print("AUC-ROC Score: ", auc_score)
        print("------------------------------------------------------------------")

        return
        
def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open(picklefilename, 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

def example_main():
    path_to_train = "part3_train.csv"
    path_to_full_train = "part3_training_data.csv"
    path_to_val = "part3_validation.csv"
    path_to_test = "part3_test.csv"
    df_full = pd.read_csv(path_to_full_train, delimiter=",")
    df_val = pd.read_csv(path_to_val, delimiter=",")
    df_test = pd.read_csv(path_to_test, delimiter=",")
    claims_raw = df_full["claim_amount"].values

    X_train = df_full
    y_train = df_full["made_claim"].values
    y_train = np.reshape(y_train, (y_train.size, 1))

    X_val = df_val
    y_val = df_val["made_claim"]

    X_test = df_test
    y_test = df_test["made_claim"]
    
    # X_train, X_test, y_train, y_test = train_test_split(df_full, y, test_size = 0.2)
    # y_train = y_train.values
    # y_train = np.reshape(y_train, (y_train.size, 1))



    pm=PricingModel(model = nn.Sequential(nn.Linear(13,10),
                                            nn.ReLU(),
                                            nn.Linear(10,8),
                                            nn.ReLU(),
                                            nn.Linear(8,6),
                                            nn.ReLU(),
                                            nn.Linear(6,4),
                                            nn.ReLU(),
                                            nn.Linear(4,1),
                                            nn.Sigmoid()), n_epochs = 100)
    pm.fit(X_train, y_train, claims_raw)
    pm.save_model()
    pm = load_model()
    print("========== validation ==========")
    pm.evaluate_architecture(X_val, y_val)
    print("============= test =============")
    pm.evaluate_architecture(X_test, y_test)


def test_main():

    classifier = load_model()
    data = pd.read_csv("part3_training_data.csv")
    X = data.drop(columns=["claim_amount", "made_claim"])
    res1 = classifier.predict_premium(X)
    res2 = classifier.predict_claim_probability(X)
    print(res1)
    print(res2)

def full_main():
    path_to_train = "part3_training_data.csv"
    df_full = pd.read_csv(path_to_train, delimiter=",")
    claims_raw = df_full["claim_amount"].values

    X_train = df_full
    y_train = df_full["made_claim"].values
    y_train = np.reshape(y_train, (y_train.size, 1))

    pm=PricingModel(model = nn.Sequential(nn.Linear(13,10),
                                            nn.ReLU(),
                                            nn.Linear(10,8),
                                            nn.ReLU(),
                                            nn.Linear(8,6),
                                            nn.ReLU(),
                                            nn.Linear(6,4),
                                            nn.ReLU(),
                                            nn.Linear(4,1),
                                            nn.Sigmoid()), n_epochs = 100)
    pm.fit(X_train, y_train, claims_raw)
    # pm.save_model()


# example_main()
# test_main()
# full_main()
