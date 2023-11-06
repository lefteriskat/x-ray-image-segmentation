import torch

class Metrics:
    '''
    Class that stores functions that return different segmentation metrics'''

    def prediction_accuracy(y_real:torch.Tensor, y_pred:torch.Tensor, segm_threshold:float=0.5)->int:
        y_pred = torch.where(y_pred > segm_threshold, 1, 0)

        return (y_pred == y_real).sum().cpu().item()



    def get_IoU(tp:int, fp:int, fn:int)->float:
        return tp/(tp+fp+fn)
        

    def get_dice_coe(y_true:torch.Tensor, y_pred_sigm:torch.Tensor, segm_threshold=0.5, epsilon=1e-3)->float:
        y_pred_mask = torch.where(y_pred_sigm>segm_threshold, 1, 0)
        num = (2*y_pred_mask*y_true).sum()
        den = (y_pred_mask+y_true).sum()
        if den==0:
            den += epsilon
        return num/den
    

    def get_tp_tn_fp_fn(y_true:torch.Tensor, y_pred_sigm:torch.Tensor, segm_threshold:float=0.5)->tuple[int]:
        y_true = y_true.cpu()
        y_true = y_true.type(torch.int64)

        y_pred = y_pred_sigm.cpu()
        y_pred = torch.where(y_pred>segm_threshold, 1, 0)
        y_pred = y_pred.type(torch.int64)

        true_positives = ((y_pred==1)&(y_true==1)).sum().item()
        true_negatives = ((y_pred==0)&(y_true==0)).sum().item()
        false_positives = ((y_pred==1)&(y_true==0)).sum().item()
        false_negatives = ((y_pred==0)&(y_true==1)).sum().item()

        return true_positives, true_negatives, false_positives, false_negatives

    
    def get_sensitivity_specificity(tp:int, tn:int, fp:int, fn:int)->tuple[float]:
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)

        return sensitivity, specificity