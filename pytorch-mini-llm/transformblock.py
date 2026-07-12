# -*- coding: utf-8 -*-
"""
"""

from torch import nn
from attention import AttentionWithRoPE
from layers import RMSNormalization,SwiGLU
    
class TransformBlock(nn.Module):
    def __init__(self,num_head,hidden_channel,input_channel, cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.num_head = num_head
        self.hidden_channel = hidden_channel
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.rmsnorm_1 = RMSNormalization(self.input_channel)
        self.rmsnorm_2 = RMSNormalization(self.input_channel)
        self.attention = AttentionWithRoPE(self.output_channel,self.num_head,cur_layer,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
        self.swiGLU = SwiGLU(self.hidden_channel,self.output_channel,alpha=alpha,lora_rank=lora_rank,use_lora=use_lora)
     
    def merge_lora_weights(self):
        self.swiGLU.merge_lora_weights_inplace()
        self.attention.merge_lora_weights_inplace()
        
    def forward(self,x,mask=None):
        res = x
        x = self.rmsnorm_1(x)
        x = self.attention(x,mask=mask)
        x = res + x
        
        res = x
        x = self.rmsnorm_2(x)
        x = self.swiGLU(x)
        x = res + x
        return x
    
