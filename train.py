import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/HousingData.csv")

CAT = ["CHAS"]
NUM = [c for c in df.columns if c not in CAT]

imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")
df[NUM] = imp_num.fit_transform(df[NUM])
df[CAT] = imp_cat.fit_transform(df[CAT])

scaler = StandardScaler()
X_num = scaler.fit_transform(df[NUM])
X = pd.concat([pd.DataFrame(X_num, columns=NUM), df[CAT].reset_index(drop=True)], axis=1)

ks = range(1, 11)
inercias = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inercias.append(km.inertia_)

plt.plot(list(ks), inercias, marker="o")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Inércia")
plt.title("Método do cotovelo")
plt.savefig("models/cotovelo.png", dpi=150)
plt.show()

K_OTIMO = 3

modelo = KMeans(n_clusters=K_OTIMO, n_init=10, random_state=42).fit(X)

joblib.dump(
    {"kmeans": modelo, "scaler": scaler, "imp_num": imp_num, "imp_cat": imp_cat,
     "NUM": NUM, "CAT": CAT},
    "models/kmeans_housing.joblib",
)
print("Modelo salvo. k =", K_OTIMO)