import torch
from torch.nn import functional as F


def jsd_loss(feat, ema_feat):
    feat += 1e-7
    ema_feat += 1e-7
    consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
    cons_loss_a = consistency_criterion(feat.log(), ema_feat.detach())
    cons_loss_b = consistency_criterion(ema_feat.log(), feat.detach())
    print(cons_loss_a, cons_loss_b)
    return cons_loss_a + cons_loss_b

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)
