from explainers.shap_explainer import shap_explain
from explainers.lime_explainer import lime_explain

def run_explanations():
    print("Running SHAP explanation...")
    shap_explain()

    print("Running LIME explanation...")
    lime_explain()

if __name__ == "__main__":
    run_explanations()
