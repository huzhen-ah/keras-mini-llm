#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:22:12 2026

@author: huzhen
"""

from torch.nn import functional as F


def pretrain_loss(input,target,pad_id):
    input = input.transpose(1,2)
    loss = F.cross_entropy(input, target,ignore_index=pad_id)
    return loss


