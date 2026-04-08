# ADAM Optimizer for SVM

from ReadFromCSV import load_data
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# -------------------- Linear SVM model --------------------
class LinearSVM(nn.Module): # this is taken from documentation
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
# SVM loss with L2 penalty
def svm_loss(model, X, y, lambda_reg=0.001):
    '''
    Computes the soft-margin SVM loss with L2 regularization.

    Loss = mean(max(0, 1 - y * f(x))) + lambda * ||w||^2

    - The hinge loss penalizes points inside the margin or misclassified.
    - The L2 term controls model complexity to prevent overfitting.
    '''
    scores = model(X) # shape(n, 1)
    hinge = torch.clamp(1 - y * scores, min=0)
    loss = torch.mean(hinge)               

    # L2 regularization on weights
    w = model.linear.weight
    loss += lambda_reg * torch.sum(w ** 2)

    return loss


def load_dataset():
    '''
    Reads data from ReadFromCSV.py
    '''
    data = load_data()
    data["target"] = np.where(data["Political Affiliation"] == "Liberal", 1, -1) # thing we want to predict

    # features and response
    X = data.drop(columns=["Political Affiliation", "target", "riding", "Constituency"]) # X is all the features except these
    X = X.select_dtypes(include=[np.number]).values
    y = data["target"].values
    return X, y

def preprocess_data(X_train, X_val, degree=3):
    '''
    Applies feature engineering and scaling:
    1. Expands features using polynomial terms to allow nonlinear decision boundaries
    2. Standardizes features (zero mean, unit variance)
    3. Converts arrays to PyTorch tensors for model training
    '''
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_val = poly.transform(X_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    return X_train, X_val, poly, scaler

def split_data(X, y, test_size=0.25, random_state=42):
    '''
    training: 75% of the data, testing: 25% of the data. Stratifying on y
    '''
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def prepare_data():
    X, y = load_dataset()
    return split_data(X, y)

def tune_hyperparameters(X, y):
    '''
    Performs grid search over hyperparameters:
    - degree: polynomial feature expansion degree
    - lr: learning rate for Adam optimizer
    - lambda_reg: L2 regularization strength
    - epochs: number of training iterations

    For each combination, 5-fold cross-validation is used to evaluate performance:
    - mean_acc: average validation accuracy across folds (measures overall model performance)
    - std_acc: standard deviation of validation accuracy (measures stability across folds)

    Selects the combination that maximizes mean cross-validated accuracy.

    '''
    degrees = [1, 2, 3]
    lrs = [0.0005, 0.001, 0.005]
    lambda_regs = [0.001, 0.01, 0.1]
    epoch_list = [300, 500, 1000]

    best_score = -np.inf
    best_params = None

    for degree in degrees:
        for lr in lrs:
            for lambda_reg in lambda_regs:
                for epochs in epoch_list:
                    print("\n-----------------------------")
                    print(f"degree={degree}, lr={lr}, lambda_reg={lambda_reg}, epochs={epochs}")

                    mean_acc, std_acc = cross_validate_svm(X, y, degree=degree, lr=lr, lambda_reg=lambda_reg, epochs=epochs, n_splits=5)

                    # results as a dictionary
                    result = {"degree": degree,"lr": lr, "lambda_reg": lambda_reg, "epochs": epochs, "mean_acc": float(mean_acc), "std_acc": float(std_acc)}
            
                    if mean_acc > best_score:
                        best_score = mean_acc
                        best_params = result

    print("\nBest Parameters:")
    print(best_params)

    return best_params

#----------- Use Stratified K-fold for Cross Validation-----------------------------
def cross_validate_svm(X, y, degree=3, lr=0.001, lambda_reg=0.1, epochs=1000, n_splits=5):
    '''
    Performs stratified k-fold cross-validation.
    - Stratification preserves the class distribution in each fold,
    which is important for imbalanced classification problems.
    - 5 folds provide a balance between computational efficiency and
    reliable performance estimation.
    '''
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        X_fold_train, X_fold_val, _, _ = preprocess_data(X_fold_train, X_fold_val, degree=degree) # preprocess within the fold

        y_fold_train = torch.tensor(y_fold_train, dtype=torch.float32).view(-1, 1)
        y_fold_val = torch.tensor(y_fold_val, dtype=torch.float32).view(-1, 1)

        model, _ = train_svm(X_fold_train,y_fold_train, lr=lr, lambda_reg=lambda_reg, epochs=epochs ) # training model

        y_val_pred = predict(model, X_fold_val)# validation predictions

        # evaluating:
        _, val_acc, _, _ = evaluate_model(y_fold_val, y_val_pred)
        fold_accuracies.append(val_acc)

        print(f"Fold {fold}: Validation Accuracy = {val_acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print("\nCV Mean Accuracy:", mean_acc)
    print("CV Std Accuracy:", std_acc)

    return mean_acc, std_acc

# -------------------- Train model --------------------
def train_svm(X_train, y_train, lr=0.001, lambda_reg=0.1, epochs=1000):
    '''
    Trains a linear SVM using the Adam optimizer.
    - Minimizes hinge loss with L2 regularization
    - Updates model parameters via gradient descent
    - Tracks loss over epochs for convergence analysis
    '''
    input_dim = X_train.shape[1]
    model = LinearSVM(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = svm_loss(model, X_train, y_train, lambda_reg=lambda_reg)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

    return model, losses

# -------------------- Predict --------------------
def predict(model, X):
    with torch.no_grad():
        scores = model(X)
        y_pred = torch.sign(scores)
        y_pred[y_pred == 0] = 1

    return y_pred.numpy().flatten()

# -------------------- Evaluate --------------------
def evaluate_model(y_true_tensor, y_pred):
    y_true = y_true_tensor.numpy().flatten()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return y_true, acc, cm, report


# -------------------- Plot loss --------------------
def plot_losses(losses):
    plt.plot(losses, color="hotpink")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Adam Optimizer Convergence")
    plt.show()

# -------------------- Display results --------------------
def display_results(train_acc, test_acc, test_cm, test_report):
    print("\nTraining Accuracy:", train_acc)
    print("Testing Accuracy:", test_acc)

    print("\nConfusion Matrix:")
    print(test_cm)

    print("\nClassification Report:")
    print(test_report)


# -------------------- Running Everything --------------------
def run_svm(X_train_raw, X_test_raw, y_train, y_test, degree, lr, lambda_reg, epochs):
    '''
    1. Preprocesses training and test data
    2. Trains the model using selected hyperparameters
    3. Generates predictions
    4. Evaluates performance (accuracy, confusion matrix, report)
    5. Plots training loss
    '''
    X_train, X_test, _, _ = preprocess_data(X_train_raw, X_test_raw, degree=degree)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model, losses = train_svm(X_train, y_train, lr=lr, lambda_reg=lambda_reg, epochs=epochs)

    y_train_pred = predict(model, X_train)
    y_test_pred = predict(model, X_test)

    _, train_acc, _, _ = evaluate_model(y_train, y_train_pred)
    _, test_acc, test_cm, test_report = evaluate_model(y_test, y_test_pred)

    display_results(train_acc, test_acc, test_cm, test_report)
    plot_losses(losses)

    return model, losses, train_acc, test_acc

def main():
    X, y = load_dataset()

    best_params = tune_hyperparameters(X, y)
    print("Chosen hyperparameters:", best_params)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return run_svm(X_train, X_test, y_train, y_test, degree=best_params["degree"], lr=best_params["lr"], lambda_reg=best_params["lambda_reg"], epochs=best_params["epochs"])

main()

