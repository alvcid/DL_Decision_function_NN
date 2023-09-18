import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib.colors import LogNorm
from sklearn.metrics import f1_score
from tensorflow.keras import models
from tensorflow.keras import layers


# Functions
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        df, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# 1. Read data
df = pd.read_csv(r"C:\Users\alvar\Documents\Udemy\2_Curso_ Aprende Inteligencia Artificial y Deep Learning con Python &\10. Limite+de+decisión+de+una+RNA+profunda.ipynb\creditcard.csv")

# 2. Visualize data
print(df.head())

print("Número de características:", len(df.columns))
print("Longitud del conjunto de datos:", len(df))

print(df["Class"].value_counts())

# See features
plt.figure(figsize=(14, 6))
plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")
plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
plt.show()

# 3. Preparation of the data
df = df.drop(["Time", "Amount"], axis=1)

# 4. Divide data
train_set, val_set, test_set = train_val_test_split(df)

X_train, y_train = remove_labels(train_set, 'Class')
X_val, y_val = remove_labels(val_set, 'Class')
X_test, y_test = remove_labels(test_set, 'Class')

# 5. 2 dimensions NN
X_train_reduced = X_train[["V10", "V14"]].copy()
X_val_reduced = X_val[["V10", "V14"]].copy()
X_test_reduced = X_test[["V10", "V14"]].copy()

X_train_reduced

# Model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train_reduced.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))#, input_shape=(X_train_reduced.shape[1],)))
model.add(layers.Dense(1, activation='sigmoid'))#, input_shape=(X_train_reduced.shape[1],)))

model.compile(optimizer='sgd',
             loss='binary_crossentropy',
             metrics=['accuracy', 'Precision'])

print(model.summary())

history = model.fit(X_train_reduced,
                   y_train,
                   epochs=30,
                   validation_data=(X_val_reduced, y_val))

# Decision boundary visualization
def plot_ann_decision_boundary(X, y, model, steps=1000):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1

    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                         np.linspace(mins[1], maxs[1], 1000))

    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = labels.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.5)

    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'k.', markersize=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'r.', markersize=2)

    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)

plt.figure(figsize=(12, 7))
plot_ann_decision_boundary(X_train_reduced.values, y_train, model)
plt.show()

y_pred = model.predict(X_train_reduced).round(0)

plt.figure(figsize=(12, 7))
plt.plot(X_train_reduced[y_pred==1]["V10"], X_train_reduced[y_pred==1]["V14"], 'go', markersize=4)
plot_ann_decision_boundary(X_train_reduced.values, y_train, model)
plt.show()

# Predict
y_pred = model.predict(X_test_reduced).round(0)

print("F1 Score:", f1_score(y_test, y_pred))