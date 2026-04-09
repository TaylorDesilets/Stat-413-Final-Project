import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import sigmoid_kernel



def load_data():
    file_path = "proj2026Dataset.csv"   
    data = pd.read_csv(file_path, encoding="latin1")
    return data


def load_dataset():
    data = load_data()

    # Liberal = 1, Non-Liberal = -1
    data["target"] = np.where(data["Political Affiliation"] == "Liberal", 1, -1)

    X = data.drop(columns=[
        "Political Affiliation",
        "target",
        "riding",
        "Constituency"
    ], errors="ignore")

    X = X.select_dtypes(include=[np.number]).values
    y = data["target"].values

    return X, y


def split_data(X, y, test_size=0.25, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def plot_conf_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Not Liberal", "Liberal"])
    plt.yticks(tick_marks, ["Not Liberal", "Liberal"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_convergence(losses, title="Convergence Plot", xlabel="Iteration"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_numpy(y_true, y_pred, method_name="Method", show_plot=True):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n=== {method_name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    if show_plot:
        plot_conf_matrix(cm, title=f"Confusion Matrix ({method_name})")

    return acc, cm, report


# RMSPROP 

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def svm_loss_torch(model, X, y, lambda_reg=0.001):
    scores = model(X)
    hinge = torch.clamp(1 - y * scores, min=0)
    loss = torch.mean(hinge)
    w = model.linear.weight
    loss += lambda_reg * torch.sum(w ** 2)
    return loss


def preprocess_linear_torch(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    return X_train_tensor, X_val_tensor, scaler


def preprocess_sigmoid_kernel_torch(X_train, X_val, gamma=0.1, coef0=0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    K_train = sigmoid_kernel(X_train_scaled, X_train_scaled, gamma=gamma, coef0=coef0)
    K_val = sigmoid_kernel(X_val_scaled, X_train_scaled, gamma=gamma, coef0=coef0)

    K_train_tensor = torch.tensor(K_train, dtype=torch.float32)
    K_val_tensor = torch.tensor(K_val, dtype=torch.float32)

    return K_train_tensor, K_val_tensor, scaler


def train_svm_rmsprop(model, X_train, y_train, lr=0.001, lambda_reg=0.001, epochs=500):
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = svm_loss_torch(model, X_train, y_train, lambda_reg=lambda_reg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def predict_torch_model(model, X):
    with torch.no_grad():
        scores = model(X).view(-1)
        y_pred = torch.where(scores >= 0, 1, -1)
    return y_pred.cpu().numpy()


def cross_validate_rmsprop(X, y, kernel_type="sigmoid",
                           lr=0.001, lambda_reg=0.001, epochs=500,
                           gamma=0.1, coef0=0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        if kernel_type == "linear":
            X_fold_train, X_fold_val, _ = preprocess_linear_torch(X_fold_train, X_fold_val)
        else:
            X_fold_train, X_fold_val, _ = preprocess_sigmoid_kernel_torch(
                X_fold_train, X_fold_val, gamma=gamma, coef0=coef0
            )

        y_fold_train = torch.tensor(y_fold_train, dtype=torch.float32).view(-1, 1)

        model = LinearSVM(X_fold_train.shape[1])
        model, _ = train_svm_rmsprop(
            model, X_fold_train, y_fold_train,
            lr=lr, lambda_reg=lambda_reg, epochs=epochs
        )

        y_val_pred = predict_torch_model(model, X_fold_val)
        val_acc = accuracy_score(y_fold_val, y_val_pred)
        fold_accuracies.append(val_acc)

    return np.mean(fold_accuracies), np.std(fold_accuracies)


def tune_hyperparameters_rmsprop(X, y):
    lrs = [0.0005, 0.001]
    lambda_regs = [0.001, 0.01]
    epoch_list = [300, 500]
    gammas = [0.01, 0.1]
    coef0_list = [0, 1]

    best_score = -np.inf
    best_params = None
    results = []

    for lr in lrs:
        for lambda_reg in lambda_regs:
            for epochs in epoch_list:
                for gamma in gammas:
                    for coef0 in coef0_list:
                        mean_acc, std_acc = cross_validate_rmsprop(
                            X, y,
                            kernel_type="sigmoid",
                            lr=lr,
                            lambda_reg=lambda_reg,
                            epochs=epochs,
                            gamma=gamma,
                            coef0=coef0,
                            n_splits=5
                        )

                        result = {
                            "lr": lr,
                            "lambda_reg": lambda_reg,
                            "epochs": epochs,
                            "gamma": gamma,
                            "coef0": coef0,
                            "cv_mean_acc": float(mean_acc),
                            "cv_std_acc": float(std_acc)
                        }
                        results.append(result)

                        if mean_acc > best_score:
                            best_score = mean_acc
                            best_params = result

    results_df = pd.DataFrame(results).sort_values("cv_mean_acc", ascending=False)
    return best_params, results_df


def run_rmsprop_final(X_train_raw, X_test_raw, y_train, y_test, params):
    X_train, X_test, _ = preprocess_sigmoid_kernel_torch(
        X_train_raw, X_test_raw,
        gamma=params["gamma"],
        coef0=params["coef0"]
    )

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = LinearSVM(X_train.shape[1])

    start = time.time()
    model, losses = train_svm_rmsprop(
        model,
        X_train,
        y_train_tensor,
        lr=params["lr"],
        lambda_reg=params["lambda_reg"],
        epochs=params["epochs"]
    )
    runtime = time.time() - start

    y_train_pred = predict_torch_model(model, X_train)
    y_test_pred = predict_torch_model(model, X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc, cm, report = evaluate_numpy(y_test, y_test_pred, method_name="RMSProp", show_plot=True)

    plot_convergence(losses, title="RMSProp Optimizer Convergence", xlabel="Epoch")

    return {
        "method": "RMSProp",
        "best_params": params,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "conf_matrix": cm,
        "report": report,
        "runtime_sec": runtime,
        "losses": losses
    }

# BFGS

def preprocess_poly_numpy(X_train, X_val, degree=3):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)

    return X_train_scaled, X_val_scaled, poly, scaler


def softplus_stable(z):
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def sigmoid_stable(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out


def svm_objective_smooth(params, X, y, lambda_reg=0.01, alpha=20.0):
    w = params[:-1]
    b = params[-1]

    scores = X @ w + b
    margins = 1 - y * scores
    z = alpha * margins

    smooth_hinge = softplus_stable(z) / alpha
    loss = np.mean(smooth_hinge) + lambda_reg * np.sum(w ** 2)
    return loss


def svm_gradient_smooth(params, X, y, lambda_reg=0.01, alpha=20.0):
    w = params[:-1]
    b = params[-1]

    scores = X @ w + b
    margins = 1 - y * scores
    z = alpha * margins

    sigma = sigmoid_stable(z)

    grad_w = -(X.T @ (sigma * y)) / X.shape[0] + 2 * lambda_reg * w
    grad_b = -np.mean(sigma * y)

    return np.append(grad_w, grad_b)


def train_svm_bfgs(X_train, y_train, lambda_reg=0.01, maxiter=200):
    n_features = X_train.shape[1]
    init_params = np.zeros(n_features + 1)
    loss_history = []

    def callback(params):
        loss = svm_objective_smooth(params, X_train, y_train, lambda_reg=lambda_reg)
        loss_history.append(loss)

    result = minimize(
        fun=svm_objective_smooth,
        x0=init_params,
        args=(X_train, y_train, lambda_reg),
        jac=svm_gradient_smooth,
        method="BFGS",
        callback=callback,
        options={"maxiter": maxiter, "disp": False}
    )

    final_loss = svm_objective_smooth(result.x, X_train, y_train, lambda_reg=lambda_reg)
    if len(loss_history) == 0 or abs(loss_history[-1] - final_loss) > 1e-12:
        loss_history.append(final_loss)

    return result, loss_history


def predict_bfgs(params, X):
    w = params[:-1]
    b = params[-1]
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)


def cross_validate_bfgs(X, y, degree=3, lambda_reg=0.01, maxiter=200, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        X_fold_train, X_fold_val, _, _ = preprocess_poly_numpy(
            X_fold_train, X_fold_val, degree=degree
        )

        result, _ = train_svm_bfgs(
            X_fold_train, y_fold_train,
            lambda_reg=lambda_reg,
            maxiter=maxiter
        )

        y_val_pred = predict_bfgs(result.x, X_fold_val)
        val_acc = accuracy_score(y_fold_val, y_val_pred)
        fold_accuracies.append(val_acc)

    return np.mean(fold_accuracies), np.std(fold_accuracies)


def tune_hyperparameters_bfgs(X, y):
    degrees = [1, 2, 3]
    lambda_regs = [0.001, 0.01, 0.1]
    maxiters = [200]

    best_score = -np.inf
    best_params = None
    results = []

    for degree in degrees:
        for lambda_reg in lambda_regs:
            for maxiter in maxiters:
                mean_acc, std_acc = cross_validate_bfgs(
                    X, y,
                    degree=degree,
                    lambda_reg=lambda_reg,
                    maxiter=maxiter,
                    n_splits=5
                )

                result = {
                    "degree": degree,
                    "lambda_reg": lambda_reg,
                    "maxiter": maxiter,
                    "cv_mean_acc": float(mean_acc),
                    "cv_std_acc": float(std_acc)
                }
                results.append(result)

                if mean_acc > best_score:
                    best_score = mean_acc
                    best_params = result

    results_df = pd.DataFrame(results).sort_values("cv_mean_acc", ascending=False)
    return best_params, results_df


def run_bfgs_final(X_train_raw, X_test_raw, y_train, y_test, params):
    X_train, X_test, _, _ = preprocess_poly_numpy(
        X_train_raw, X_test_raw, degree=params["degree"]
    )

    start = time.time()
    result, losses = train_svm_bfgs(
        X_train, y_train,
        lambda_reg=params["lambda_reg"],
        maxiter=params["maxiter"]
    )
    runtime = time.time() - start

    y_train_pred = predict_bfgs(result.x, X_train)
    y_test_pred = predict_bfgs(result.x, X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc, cm, report = evaluate_numpy(y_test, y_test_pred, method_name="BFGS", show_plot=True)

    plot_convergence(losses, title="BFGS Optimizer Convergence", xlabel="Iteration")

    return {
        "method": "BFGS",
        "best_params": params,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "conf_matrix": cm,
        "report": report,
        "runtime_sec": runtime,
        "losses": losses
    }


# NELDER-MEAD SECTION
def svm_objective_nm(params, X, y, lambda_reg=0.01):
    w = params[:-1]
    b = params[-1]

    scores = X @ w + b
    margins = 1 - y * scores
    hinge = np.maximum(0, margins)

    loss = np.mean(hinge) + lambda_reg * np.sum(w ** 2)
    return loss


def train_svm_nelder_mead(X_train, y_train, lambda_reg=0.01, maxiter=1000):
    n_features = X_train.shape[1]
    init_params = np.zeros(n_features + 1)

    loss_history = []

    def callback(params):
        loss = svm_objective_nm(params, X_train, y_train, lambda_reg=lambda_reg)
        loss_history.append(loss)

    result = minimize(
        fun=svm_objective_nm,
        x0=init_params,
        args=(X_train, y_train, lambda_reg),
        method="Nelder-Mead",
        callback=callback,
        options={"maxiter": maxiter, "disp": False, "xatol": 1e-4, "fatol": 1e-4}
    )

    final_loss = svm_objective_nm(result.x, X_train, y_train, lambda_reg=lambda_reg)
    if len(loss_history) == 0 or abs(loss_history[-1] - final_loss) > 1e-12:
        loss_history.append(final_loss)

    return result, loss_history


def predict_nm(params, X):
    w = params[:-1]
    b = params[-1]
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)


def cross_validate_nelder_mead(X, y, degree=2, lambda_reg=0.01, maxiter=1000, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        X_fold_train, X_fold_val, _, _ = preprocess_poly_numpy(
            X_fold_train, X_fold_val, degree=degree
        )

        result, _ = train_svm_nelder_mead(
            X_fold_train,
            y_fold_train,
            lambda_reg=lambda_reg,
            maxiter=maxiter
        )

        y_val_pred = predict_nm(result.x, X_fold_val)
        val_acc = accuracy_score(y_fold_val, y_val_pred)
        fold_accuracies.append(val_acc)

    return np.mean(fold_accuracies), np.std(fold_accuracies)


def tune_hyperparameters_nelder_mead(X, y):
    degrees = [1, 2]
    lambda_regs = [0.001, 0.01, 0.1]
    maxiters = [500, 1000]

    best_score = -np.inf
    best_params = None
    results = []

    for degree in degrees:
        for lambda_reg in lambda_regs:
            for maxiter in maxiters:
                mean_acc, std_acc = cross_validate_nelder_mead(
                    X, y,
                    degree=degree,
                    lambda_reg=lambda_reg,
                    maxiter=maxiter,
                    n_splits=5
                )

                result = {
                    "degree": degree,
                    "lambda_reg": lambda_reg,
                    "maxiter": maxiter,
                    "cv_mean_acc": float(mean_acc),
                    "cv_std_acc": float(std_acc)
                }
                results.append(result)

                if mean_acc > best_score:
                    best_score = mean_acc
                    best_params = result

    results_df = pd.DataFrame(results).sort_values("cv_mean_acc", ascending=False)
    return best_params, results_df


def run_nelder_mead_final(X_train_raw, X_test_raw, y_train, y_test, params):
    X_train, X_test, _, _ = preprocess_poly_numpy(
        X_train_raw, X_test_raw, degree=params["degree"]
    )

    start = time.time()
    result, losses = train_svm_nelder_mead(
        X_train,
        y_train,
        lambda_reg=params["lambda_reg"],
        maxiter=params["maxiter"]
    )
    runtime = time.time() - start

    y_train_pred = predict_nm(result.x, X_train)
    y_test_pred = predict_nm(result.x, X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc, cm, report = evaluate_numpy(y_test, y_test_pred, method_name="Nelder-Mead", show_plot=True)

    plot_convergence(losses, title="Nelder-Mead Optimizer Convergence", xlabel="Iteration")

    return {
        "method": "Nelder-Mead",
        "best_params": params,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "conf_matrix": cm,
        "report": report,
        "runtime_sec": runtime,
        "losses": losses
    }



def plot_all_convergence(rmsprop_losses, bfgs_losses, nm_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(rmsprop_losses, label="RMSProp")
    plt.plot(bfgs_losses, label="BFGS")
    plt.plot(nm_losses, label="Nelder-Mead")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Comparison of All Three Optimizers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n================ RMSPROP TUNING ================\n")
    best_rmsprop, rmsprop_tuning = tune_hyperparameters_rmsprop(X, y)
    rmsprop_tuning.to_csv("rmsprop_tuning_results.csv", index=False)
    print("Best RMSProp parameters:", best_rmsprop)

    print("\n================ BFGS TUNING ================\n")
    best_bfgs, bfgs_tuning = tune_hyperparameters_bfgs(X, y)
    bfgs_tuning.to_csv("bfgs_tuning_results.csv", index=False)
    print("Best BFGS parameters:", best_bfgs)

    print("\n================ NELDER-MEAD TUNING ================\n")
    best_nm, nm_tuning = tune_hyperparameters_nelder_mead(X, y)
    nm_tuning.to_csv("nelder_mead_tuning_results.csv", index=False)
    print("Best Nelder-Mead parameters:", best_nm)

    print("\n================ RMSPROP ================\n")
    rmsprop_results = run_rmsprop_final(X_train, X_test, y_train, y_test, best_rmsprop)

    print("\n================ BFGS ================\n")
    bfgs_results = run_bfgs_final(X_train, X_test, y_train, y_test, best_bfgs)

    print("\n================ NELDER-MEAD ================\n")
    nm_results = run_nelder_mead_final(X_train, X_test, y_train, y_test, best_nm)

    comparison_df = pd.DataFrame([
        {
            "Method": rmsprop_results["method"],
            "CV Mean Accuracy": best_rmsprop["cv_mean_acc"],
            "CV Std Accuracy": best_rmsprop["cv_std_acc"],
            "Train Accuracy": rmsprop_results["train_acc"],
            "Test Accuracy": rmsprop_results["test_acc"],
            "Runtime (sec)": rmsprop_results["runtime_sec"]
        },
        {
            "Method": bfgs_results["method"],
            "CV Mean Accuracy": best_bfgs["cv_mean_acc"],
            "CV Std Accuracy": best_bfgs["cv_std_acc"],
            "Train Accuracy": bfgs_results["train_acc"],
            "Test Accuracy": bfgs_results["test_acc"],
            "Runtime (sec)": bfgs_results["runtime_sec"]
        },
        {
            "Method": nm_results["method"],
            "CV Mean Accuracy": best_nm["cv_mean_acc"],
            "CV Std Accuracy": best_nm["cv_std_acc"],
            "Train Accuracy": nm_results["train_acc"],
            "Test Accuracy": nm_results["test_acc"],
            "Runtime (sec)": nm_results["runtime_sec"]
        }
    ])

    comparison_df = comparison_df.sort_values("Test Accuracy", ascending=False)
    comparison_df.to_csv("optimizer_comparison_summary.csv", index=False)

    print("\n================ COMPARISON TABLE ================\n")
    print(comparison_df.to_string(index=False))

    plot_all_convergence(
        rmsprop_results["losses"],
        bfgs_results["losses"],
        nm_results["losses"]
    )


if __name__ == "__main__":
    main()