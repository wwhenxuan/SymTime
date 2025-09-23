from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_short_term_forecasting import Exp_Short_Term_Forecast
from .exp_classification import Exp_Classification
from .exp_imputation import Exp_Imputation
from .exp_anomaly_detection import Exp_Anomaly_Detection

from .exp_pretraining import Exp_Pretraining


def Exp(task_name):
    """Choose the right exp for model training"""
    if task_name == "long_term_forecast":
        return Exp_Long_Term_Forecast
    elif task_name == "short_term_forecast":
        return Exp_Short_Term_Forecast
    elif task_name == "classification":
        return Exp_Classification
    elif task_name == "imputation":
        return Exp_Imputation
    elif task_name == "anomaly_detection":
        return Exp_Anomaly_Detection
    else:
        raise ValueError("Invalid task name")
