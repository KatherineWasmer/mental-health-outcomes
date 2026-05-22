import pandas as pd 
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

wh_data = pd.read_csv("imputed_data.csv")

def train_model():
    # process data and split into testing and training
    data = wh_data
    X = data.drop(columns="mental_health_prevalence")
    y = data["mental_health_prevalence"]
    
    # separate numerical and categorical columns 
    X_num = X.select_dtypes(include="number")
    X_cat = X.select_dtypes(exclude="number")
    num_attribs = list(X_num.columns)
    cat_attribs = list(X_cat.columns)

    # encode and scale data 
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
    ]) 

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=5)),
        ("model", SVC(gamma=2, C=1))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "mental-health-pipeline.pkl")
    print("Pipeline processed")
    return pipeline 

train_model()