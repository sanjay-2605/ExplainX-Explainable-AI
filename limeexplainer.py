from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch
from models.model import SimpleNN
from utils.data_loader import load_data

def lime_explain():
    X_train, X_test, y_train, y_test = load_data()

    model = SimpleNN(X_train.shape[1])
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    explainer = LimeTabularExplainer(X_train, mode='classification')

    exp = explainer.explain_instance(
        X_test[0],
        lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    )

    exp.show_in_notebook()

if __name__ == "__main__":
    lime_explain()
