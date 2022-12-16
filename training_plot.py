import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import learning_curve
import pandas as pd

df = pd.read_csv('history1.csv')
# df2 = pd.read_csv('history2.csv')
# df = pd.concat([df, df2], ignore_index=True)
fig = plt.figure(figsize=(8, 6))
key = ['value_loss', 'policy_loss', 'policy_sparse_categorical_accuracy']
y_label = ["MAE", "Sparse Categorical\nCross Entropy", "Sparse Categorical\nAccuracy"]
subtitle = ["Value Loss", "Policy Loss", "Policy Accuracy"]

for i in range(3):
    ax = fig.add_subplot(3, 1, i+1)
    ax.plot(df[key[i]].str.strip('[]').astype(float), label="Training score")
    # plt.fill_between(mean[i] - std[i], mean[i] + std[i], color="#DDDDDD")
    plt.ylabel(y_label[i])
    if i == 2:
        plt.xlabel("Epoch")
    ax.title.set_text(subtitle[i])
plt.tight_layout()
plt.show()
