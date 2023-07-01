import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class DOTLoss(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(DOTLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, labels):
        # DOT loss calculation
        batch_size, num_classes = pred.size()
        ones = torch.eye(num_classes, device=pred.device)
        targets_2d = torch.index_select(ones, dim=0, index=labels)
        loss = torch.sum(-targets_2d * pred, dim=1)

        # Return the loss
        return loss.mean()


# def DOTLoss()
#     return loss = torch.sum(- targets_2d * pred, 1)

# def compute_loss(targets, pred, type='ce', reduction='mean', weight=None, cls_weight=None, rce_weight=None, soft_flag=False):
#     batch_size, num_cls = pred.size()
#     if targets.ndim < 2:
#         ones = torch.eye(num_cls, device=pred.device)
#         targets_2d = torch.index_select(ones, dim=0, index=targets)
#         targets_1d = targets
#         if soft_flag: # no soft label to use
#             print('Warning: no soft_label provided')
#     else:
#         targets_2d = targets
#         targets_1d = targets_2d.argmax(dim=1)
#         if not soft_flag:  # use hard label
#             ones = torch.eye(num_cls, device=pred.device)
#             targets_2d = torch.index_select(ones, dim=0, index=targets_1d)

#     if type == 'ce':
#         loss = cross_entropy(targets_2d, pred, reduction=reduction)
#     elif type == 'sce':
#         loss_criterion = SCELoss()
#         loss = loss_criterion(targets_2d, pred, reduction=reduction, rce_weight = rce_weight)
#     elif type == 'dot' or type == 'dot_d':
#         loss = torch.sum(- targets_2d * pred, 1)

#     if weight is not None:
#         loss = loss * weight
#     if cls_weight is not None: 
#         elementwise_weight = cls_weight[targets_1d]
#         loss = loss * elementwise_weight
        
#     loss = loss.sum() / batch_size
#     return loss