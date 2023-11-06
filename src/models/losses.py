
import torch
import torch.nn.functional as F


class Losses:
    '''
    Class that holds loss functions that can be used to train segmentation models
    '''
    def focal_loss(y_real:torch.Tensor, y_pred:torch.Tensor)->torch.Tensor:
        ### y_real and y_pred is [batch_n, channels=1, h, w]: [6, 1, 128, 128]
        y_real_flat = y_real.view(y_real.size(0), -1)
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        
        gamma = 2 # a good value from the paper of Lin
        weight = (1-F.sigmoid(y_pred_flat)).pow(gamma)
        tmp = weight*y_real_flat*torch.log(F.sigmoid(y_pred_flat)) + (1-y_real_flat)*torch.log(1-F.sigmoid(y_pred_flat))
        return -torch.mean(tmp)
    
    def bce_loss(y_real:torch.Tensor, y_pred:torch.Tensor)->torch.Tensor:
        y_pred = torch.clip(y_pred, -10, 10)
        return torch.mean(y_pred - y_real * y_pred + torch.log(1 + torch.exp(-y_pred)))
    
    def dice_loss(y_real:torch.Tensor, y_pred:torch.Tensor)->torch.Tensor:
        ### y_real and y_pred is [batch_n, channels=1, h, w]: [6, 1, 128, 128]
        y_real_flat = y_real.view(y_real.size(0), -1)
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        num = (2 * y_real_flat * F.sigmoid(y_pred_flat) + 1).mean()
        den = (y_real_flat + F.sigmoid(y_pred_flat)).mean() + 1
        return 1 - (num / den)
    
    def bce_total_variation(y_real:torch.Tensor, y_pred:torch.Tensor, lambda_:float=0.1)->torch.Tensor:

        def total_variation_term():        
            y_pred_x = y_pred[:,:,:-1,:]
            y_pred_xp1 = y_pred[:,:,1:,:]
            
            y_pred_y = y_pred[:,:,:,:-1]
            y_pred_yp1 = y_pred[:,:,:,1:]
            
            y_pred_x_flat = torch.flatten(y_pred_x,start_dim=1)
            y_pred_xp1_flat = torch.flatten(y_pred_xp1,start_dim=1)
            
            y_pred_y_flat = torch.flatten(y_pred_y,start_dim=1)
            y_pred_yp1_flat = torch.flatten(y_pred_yp1,start_dim=1)
            
            term1 = torch.sum( torch.abs(F.sigmoid(y_pred_xp1_flat) - F.sigmoid(y_pred_x_flat)) )
            term2 = torch.sum( torch.abs(F.sigmoid(y_pred_yp1_flat) - F.sigmoid(y_pred_y_flat)) )
            return term1 + term2

        return Losses.bce_loss(y_real, y_pred) + lambda_*total_variation_term()