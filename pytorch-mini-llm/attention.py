#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:06:29 2026

@author: huzhen
"""
import torch
from torch import nn
from torch.nn import functional as F
from rope import rope_exp


class AttentionWithRoPE(nn.Module):
    def __init__(self,input_channel,num_head,cur_layer,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = input_channel
        self.num_head = num_head
        self.cur_layer = cur_layer
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.use_lora = use_lora
        self.q_dense = nn.Linear(self.input_channel,self.output_channel)
        self.k_dense = nn.Linear(self.input_channel,self.output_channel)
        self.v_dense = nn.Linear(self.input_channel,self.output_channel)
        self.out_dense = nn.Linear(self.input_channel,self.output_channel)

        self.lora_q_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_q_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_q_B.weight)

        self.lora_k_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_k_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_k_B.weight)
        
        self.lora_v_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_v_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_v_B.weight)
        
        self.lora_out_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_out_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_out_B.weight)
        

    def merge_lora_weights_inplace(self):
        with torch.no_grad():
            lora_q_delat_weight = self.scale * torch.matmul(self.lora_q_B.weight,self.lora_q_A.weight)
            self.q_dense.weight.add_(lora_q_delat_weight)
            
            lora_k_delta_weight = self.scale *  torch.matmul(self.lora_k_B.weight,self.lora_k_A.weight)
            self.k_dense.weight.add_(lora_k_delta_weight)
            
            lora_v_delta_weight = self.scale * torch.matmul(self.lora_v_B.weight,self.lora_v_A.weight)
            self.v_dense.weight.add_(lora_v_delta_weight)
            
            lora_out_delta_weight = self.scale * torch.matmul(self.lora_out_B.weight,self.lora_out_A.weight)
            self.out_dense.weight.add_(lora_out_delta_weight)
        
    def reshape_and_permute(self,x):
        b,t,s = x.shape
        
        x = x.reshape(-1,t,self.num_head,s//self.num_head)
        x = x.permute(0,2,1,3)
        return x

    def forward(self,x,mask=None):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)
        if self.use_lora:
            q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
            k = k + self.scale * self.lora_k_B(self.lora_k_A(x))
            v = v + self.scale * self.lora_v_B(self.lora_v_A(x))

        b,t,s = q.shape

        q = self.reshape_and_permute(q)
        k = self.reshape_and_permute(k)
        v = self.reshape_and_permute(v)


        q = rope_exp(q)
        k = rope_exp(k)
        qk = torch.einsum("bhms,bhns->bhmn", q,k)

        mask_q = torch.arange(qk.size(2),dtype=torch.int32,device=qk.device)[:,None]#(m,1)
        mask_k = torch.arange(qk.size(3),dtype=torch.int32,device=qk.device)#(n)
        causal_mask = (mask_q < mask_k).to(torch.float32)#(m,n)
        if mask is not None:#mask: (batch, key_len), 1 表示 pad key
            causal_mask = causal_mask[None,None,:,:]
            mask = mask[:,None,None,:]
            mask = torch.maximum(causal_mask, mask)
            mask = mask * (-1e10)
            qk = qk + mask
        else:
            qk = qk - causal_mask*1e10

        score = F.softmax(qk/(s//self.num_head)**0.5,dim=-1)

        _out = torch.einsum("bhmn,bhns->bhms", score,v)
        _out = _out.permute(0,2,1,3)
        _out = _out.reshape(b,t,s)
        out = self.out_dense(_out)
        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(_out))
        return out


