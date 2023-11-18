import logging

import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_specificity, multiclass_f1_score, multiclass_jaccard_index

logger = logging.getLogger(__name__)


class Metrics2:
    @staticmethod
    def get_accuracy(y_pred, mask):
        return multiclass_accuracy(y_pred, mask, num_classes=3)
    
    @staticmethod
    def get_specificity(y_pred, mask):
        return multiclass_specificity(y_pred, mask, num_classes=3)
    
    @staticmethod
    def get_iou(y_pred, mask):  #iou
        return multiclass_jaccard_index(y_pred, mask, num_classes=3)
    
    @staticmethod
    def get_dice_coef(y_pred, mask):
        return multiclass_f1_score(y_pred, mask, num_classes=3)
    
    @staticmethod
    def get_all_metrics(y_pred, mask):
        return {"accuracy": multiclass_accuracy(y_pred, mask, num_classes=3),
                "specificity": multiclass_specificity(y_pred, mask, num_classes=3), 
                "iou": multiclass_jaccard_index(y_pred, mask, num_classes=3),
                "dice": multiclass_f1_score(y_pred, mask, num_classes=3)}
        
class Metrics:
    """
    Class that stores functions that return different segmentation metrics"""

    @staticmethod
    def get_predicted_segmentation_mask(y_pred):
        predictions = torch.nn.functional.softmax(y_pred, dim=1)
        pred_mask = torch.argmax(predictions, dim=1)
        pred_mask = pred_mask.long()
        return pred_mask

    def prediction_accuracy(
        y_real: torch.Tensor, y_pred: torch.Tensor, segm_threshold: float = 0.5
    ) -> int:
        y_pred = Metrics.get_predicted_segmentation_mask(y_pred)
        batch_size, height, width = y_pred.shape
        return (y_pred == y_real).sum().cpu().item() / (batch_size * height * width)

    def get_IoU(tp: int, fp: int, fn: int) -> float:
        return tp / (tp + fp + fn)

    def get_dice_coe(
        y_true: torch.Tensor, y_pred: torch.Tensor, segm_threshold=0.5, epsilon=1e-3
    ) -> float:
        y_pred_mask = Metrics.get_predicted_segmentation_mask(y_pred)
        num = (2 * y_pred_mask * y_true).sum()
        den = (y_pred_mask + y_true).sum()
        if den == 0:
            den += epsilon
        return num / den

    def get_tp_tn_fp_fn(
        y_true: torch.Tensor, y_pred: torch.Tensor, segm_threshold: float = 0.5
    ) -> tuple[int]:
        y_true = y_true.cpu()
        y_true = y_true.type(torch.int64)

        y_pred = y_pred.cpu()
        y_pred = Metrics.get_predicted_segmentation_mask(y_pred)
        y_pred = y_pred.type(torch.int64)

        true_positives = ((y_pred == 1) & (y_true == 1)).sum().item()
        true_negatives = ((y_pred == 0) & (y_true == 0)).sum().item()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum().item()
        false_negatives = ((y_pred == 0) & (y_true == 1)).sum().item()

        return true_positives, true_negatives, false_positives, false_negatives

    def get_sensitivity_specificity(tp: int, tn: int, fp: int, fn: int) -> tuple[float]:
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return sensitivity, specificity
