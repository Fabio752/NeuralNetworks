from imblearn.combine import SMOTEENN
import numpy as np

path_to_train = "../../part2_train_.csv"
print(path_to_train)
X_raw = np.genfromtxt(path_to_train, delimiter=',')[1:, :]

y_raw = X_raw[:, X_raw.shape[1]-1:]
X_raw = X_raw[:, :X_raw.shape[1]-1]

out = "drv_age1,vh_age,vh_cyl,vh_din,pol_bonus,vh_sale_begin,vh_sale_end,vh_value,vh_speed,claim_amount,made_claim\n"
smote = SMOTEENN()
X_sm, y_sm = smote.fit_resample(X_raw,y_raw)
for i in range(len(X_sm)):
    out += ",".join(str(x) for x in X_sm[i]) + "," + str(y_sm[i]) + "\n"
f= open("train_raw_.csv","w+")
f.write(out)
f.close()

