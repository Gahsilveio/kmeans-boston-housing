import joblib
import numpy as np
import pandas as pd

art = joblib.load("models/kmeans_housing.joblib")
kmeans, scaler = art["kmeans"], art["scaler"]
imp_num, imp_cat = art["imp_num"], art["imp_cat"]
NUM, CAT = art["NUM"], art["CAT"]


def inferir(residencia: dict):
    df = pd.DataFrame([residencia])[NUM + CAT]

    df[NUM] = imp_num.transform(df[NUM])
    df[CAT] = imp_cat.transform(df[CAT])
    X = pd.DataFrame(np.hstack([scaler.transform(df[NUM]), df[CAT].values]), columns=NUM + CAT)

    grupo = int(kmeans.predict(X)[0])

    c = kmeans.cluster_centers_[grupo]
    c_num = scaler.inverse_transform(c[: len(NUM)].reshape(1, -1))[0]
    centroide = dict(zip(NUM, c_num.round(3)))
    centroide.update(dict(zip(CAT, c[len(NUM):].round(3))))
    return grupo, centroide


if __name__ == "__main__":
    nova = {"CRIM": 0.02, "ZN": 18, "INDUS": 2.3, "CHAS": 0, "NOX": 0.54,
            "RM": 6.5, "AGE": 65, "DIS": 4.1, "RAD": 1, "TAX": 296,
            "PTRATIO": 15.3, "B": 396.9, "LSTAT": 5, "MEDV": 24}
    grupo, centroide = inferir(nova)
    print(f"A residência pertence ao grupo {grupo}")
    print("Centroide desnormalizado:")
    for k, v in centroide.items():
        print(f"  {k}: {v}")