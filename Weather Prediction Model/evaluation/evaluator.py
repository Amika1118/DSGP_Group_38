from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report
)
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, X, y):
        logger.info(f"Evaluating {model.model_name}...")

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        metrics = {
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'cohen_kappa': cohen_kappa_score(y, y_pred)
        }

        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, zero_division=0)

        logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        return {
            'metrics': metrics,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }