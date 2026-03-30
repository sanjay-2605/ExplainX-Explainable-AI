import shap
import torch
import numpy as np
from models.model import SimpleNN
from utils.data_loader import load_data

def shap_explain():
    X_train, X_test, y_train, y_test = load_data()

    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
        X_train[:100]
    )

    shap_values = explainer.shap_values(X_test[:10])
    shap.summary_plot(shap_values, X_test[:10])

if __name__ == "__main__":
    shap_explain()
