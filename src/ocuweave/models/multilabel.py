"""Independent lesion-query heads with a shared visual encoder and local refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from ocuweave.models.vision.compact import CompactEncoder
from ocuweave.models.heads.query_mask import QueryMaskHead


class MultiLabelLQNet(nn.Module):
    def __init__(self, classes=4, width=24, dim=48, queries=8, local_detail=True):
        super().__init__()
        if classes<1 or dim%4:raise ValueError('Positive classes and embedding divisible by four required')
        self.encoder=CompactEncoder(width)
        self.heads=nn.ModuleList(QueryMaskHead(self.encoder.channels,dim,queries) for _ in range(classes))
        self.local_detail=local_detail
        if local_detail:
            self.detail=nn.Sequential(nn.Conv2d(3,width,3,stride=2,padding=1),nn.GroupNorm(1,width),nn.GELU(),
                                      nn.Conv2d(width,width,3,padding=1,groups=width),nn.GroupNorm(1,width),nn.GELU())
            self.refine=nn.Sequential(nn.Conv2d(width+classes,width,1),nn.GELU(),nn.Conv2d(width,classes,1))
            nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)

    def forward(self, images):
        features=self.encoder(images)
        logits=torch.cat([head(features,images.shape[-2:]) for head in self.heads],dim=1)
        if not self.local_detail:return logits
        detail=self.detail(images)
        context=F.interpolate(logits,size=detail.shape[-2:],mode='bilinear',align_corners=False)
        return logits+F.interpolate(self.refine(torch.cat((detail,context),dim=1)),size=images.shape[-2:],mode='bilinear',align_corners=False)
