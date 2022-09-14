# import sklearn
# print(sklearn.__version__)
#
# from sklearn.neighbors import _dist_metrics

import cv2
import numpy as np
import mediapipe as mp
import time
from sklearn import preprocessing
import PoseModule as pm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load the model from disk
loaded_model = pickle.load(open('knnpickle_regions.sav', 'rb'))

# result = loaded_model.predict([test])

cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    image = detector.findPose(img,draw=False)
    className = ""
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        # Extracting the feature list
        # Predict gesture
        (row, image) = detector.regionFeatures(lmList, image, thickness=3)
        # # regions = [body, lH1, rH1, lH2, rH2]
        # row = []
        # for region in regions:
        #     for i in range(len(region)):
        #         row.append(region[i])

        row = row[2:]
        F = np.array([row])
        F.reshape(1, -1)
        print(F.shape)
        # mean_scaling = StandardScaler()
        scaler = preprocessing.StandardScaler().fit(F)
        scaled = scaler.transform(F)
        # scaled = mean_scaling.fit_transform(F)
        print(scaled)
        pca10 = PCA(n_components = 0.95)
        test = pca10.fit_transform(scaled)

        # scaler = preprocessing.StandardScaler().fit(F)
        # row = scaler.transform(F)
        prediction = loaded_model.predict(test)
        className = prediction[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == ord('q'):
        break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

