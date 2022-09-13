import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix

pose_region = pd.read_csv('pose_region_props.csv')
poselandmark = pd.read_csv('poselandmarks_props.csv')
pose_region = pose_region.drop(['b_x', 'b_y'], axis=1)

mean_scaling = StandardScaler()
pose_region_scaled = pd.DataFrame(mean_scaling.fit_transform(pose_region.iloc[:,:-1]), columns= pose_region.columns[:-1])

pca10 = PCA(n_components=0.95)
pc10_regions = pd.DataFrame(pca10.fit_transform(pose_region_scaled))
print(pc10_regions)

labels = pose_region["Pose"]
pc10_regions["Pose"] = labels
print(pc10_regions)

def create_knn_model(df):
#     df = pd.read_csv(file)
    features = df.columns
    labels = df["Pose"].tolist()

    # Train test split
    X = df.drop(columns=['Pose'])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    print(y_train[:25])

    # Creating KNN Model
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k,weights = 'distance', metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return knn

knn_model = create_knn_model(pc10_regions)


knnPickle = open('knnpickle_regions.sav', 'wb')

# source, destination
pickle.dump(knn_model, knnPickle)

pose_landmark_scaled = pd.DataFrame(mean_scaling.fit_transform(poselandmark.iloc[:,:-1]), columns= poselandmark.columns[:-1])

pc10_land = pd.DataFrame(pca10.fit_transform(pose_landmark_scaled))
pc10_land["Pose"] = labels

knn_model_land = create_knn_model(pc10_land)


knnPickleLand = open('knnpickle_land.sav', 'wb')

# source, destination
pickle.dump(knn_model_land, knnPickleLand)
