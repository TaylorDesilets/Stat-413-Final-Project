import matplotlib.pyplot as plt
import numpy as np

# confusion matrix values
cm = np.array([[36, 7],
               [13, 30]])

# class labels
labels = [-1, 1]

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Pastel2')
plt.title("Confusion Matrix for Predicting Liberal / Non Liberal")
plt.colorbar()

# tick marks
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)

# axis labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# add counts inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.tight_layout()
plt.show()