import pandas as pd 
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load our imputed data set for containerization 
data = pd.read_csv("imputed_data.csv")
X = data.drop(columns="mental_health_prevalence")
y = data["mental_health_prevalence"]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# Filter out numerical and categorical columns for proper encoding.
X_num = X.select_dtypes(include="number")
X_cat = X.select_dtypes(exclude="number")

num_attribs = list(X_num.columns)
cat_attribs = list(X_cat.columns)
all_attribs = list(X_train.columns)

combined_pipeline = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
]) 
data_processed = combined_pipeline.fit_transform(X_train)
test_data_processed = combined_pipeline.transform(X_test)

# perform a PCA to eliminate collinearity 
pca = PCA(n_components=5)
X_train_proc = pca.fit_transform(data_processed)
X_test_proc = pca.transform(test_data_processed)

model = SVC(gamma=2, C=1)
model.fit(X_train_proc, y_train)
predictions = model.predict(X_test_proc)

joblib.dump(model, 'model.pkl')
print("Model trained and saved to model.pkl")