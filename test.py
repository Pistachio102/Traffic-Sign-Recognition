from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from PIL import Image

# testing accuracy on test dataset

y_test = pd.read_csv('test.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# np.set_printoptions(threshold=100)

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)
model = load_model("my_model.h5")
pred = model.predict(X_test)
pred = [np.argmax(prediction) for prediction in pred]
pred = np.array(pred)

print(labels)
print(pred)

# Accuracy with the test data
# print(accuracy_score(labels, pred))