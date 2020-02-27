from imblearn.over_sampling import SMOTE
import numpy as np

ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
path_to_train = "../part2_train_.csv"
print(path_to_train)
X_raw = np.genfromtxt(path_to_train, delimiter=',')[1:, :]

y_raw = X_raw[:, X_raw.shape[1]-1:]
X_raw = X_raw[:, :X_raw.shape[1]-2]

for ratio in ratios:
    out = "drv_age1,vh_age,vh_cyl,vh_din,pol_bonus,vh_sale_begin,vh_sale_end,vh_value,vh_speed,claim_amount,made_claim\n"
    smote = SMOTE(sampling_strategy=ratio)
    X_sm, y_sm = smote.fit_sample(X_raw,y_raw)
    for i in range(len(X_sm)):
        out += ",".join(str(x) for x in X_sm[i]) + "," + str(y_sm[i]) + "\n"
    f= open("train_raw_" + str(ratio) + ".txt","w+")
    f.write(out)
    f.close()

