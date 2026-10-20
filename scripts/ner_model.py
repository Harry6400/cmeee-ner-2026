"""逐字符编码器与GlobalPointer；保留嵌套、交叉及同跨度不同类别。"""
from __future__ import annotations
import math
import torch
from torch import nn
from ner_core import TYPES


def configure(seed, threads=4):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GlobalPointer(nn.Module):
    def __init__(self, hidden, size=64, rope=True):
        super().__init__()
        self.proj = nn.Linear(hidden, len(TYPES)*size*2)
        self.size, self.rope = size, rope

    def forward(self, hidden, char_mask):
        batch, length, _ = hidden.shape
        q, k = self.proj(hidden).view(batch,length,len(TYPES),2,self.size).unbind(3)
        if self.rope:
            positions = torch.arange(length, device=hidden.device, dtype=hidden.dtype)
            inv = 10000 ** (-torch.arange(0,self.size,2,device=hidden.device,dtype=hidden.dtype)/self.size)
            angles = torch.outer(positions, inv)
            sin = angles.sin().repeat_interleave(2,dim=-1)[None,:,None,:]
            cos = angles.cos().repeat_interleave(2,dim=-1)[None,:,None,:]
            def rotate(x):
                pair = torch.stack((-x[...,1::2],x[...,::2]),dim=-1).flatten(-2)
                return x*cos+pair*sin
            q, k = rotate(q), rotate(k)
        logits = torch.einsum('blhd,bmhd->bhlm',q,k) / math.sqrt(self.size)
        mask = char_mask[:,None,:,None] & char_mask[:,None,None,:]
        upper = torch.ones(length,length,device=hidden.device,dtype=torch.bool).triu()
        return logits.masked_fill(~(mask & upper[None,None]), -1e12)


class Extractor(nn.Module):
    def __init__(self, method, cfg, vocab, bert_config=None, pretrained=None):
        super().__init__()
        self.method = method
        if method == 'N0':
            self.embedding = nn.Embedding(len(vocab),cfg['embedding_dim'],padding_idx=0)
            self.encoder = nn.LSTM(cfg['embedding_dim'],cfg['lstm_hidden'],batch_first=True,bidirectional=True)
            hidden = 2*cfg['lstm_hidden']
        else:
            from transformers import BertConfig, BertModel
            if pretrained:
                self.encoder = BertModel.from_pretrained(pretrained, local_files_only=True,
                                                        add_pooling_layer=False, attn_implementation='eager')
            else:
                bc = BertConfig(**bert_config)
                bc._attn_implementation = 'eager'
                self.encoder = BertModel(bc,add_pooling_layer=False)
            hidden = self.encoder.config.hidden_size
        self.head = GlobalPointer(hidden,cfg['head_size'],cfg['rope'])

    def forward(self, ids, attention, chars):
        if self.method == 'N0':
            embedded = self.embedding(ids)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, attention.sum(1).cpu(),
                                                       batch_first=True,enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=ids.shape[1])
        else:
            hidden = self.encoder(input_ids=ids,attention_mask=attention.long()).last_hidden_state
        return self.head(hidden, chars)


def loss_fn(logits, labels):
    # 多标签categorical cross-entropy，按每个样本/类别的全部合法跨度计算。
    x, y = logits.flatten(2), labels.flatten(2)
    signed = (1-2*y)*x
    negative = signed.masked_fill(y.bool(), -1e12)
    positive = signed.masked_fill(~y.bool(), -1e12)
    zeros = torch.zeros_like(x[...,:1])
    return (torch.logsumexp(torch.cat((negative,zeros),-1),-1) +
            torch.logsumexp(torch.cat((positive,zeros),-1),-1)).mean()


def collate(items, vocab, method, device='cpu', labels=True):
    offset = 1 if method=='N1' else 0
    length = max(len(w['text']) for w in items) + 2*offset
    ids = torch.zeros(len(items),length,dtype=torch.long)
    attention = torch.zeros_like(ids,dtype=torch.bool)
    chars = torch.zeros_like(ids,dtype=torch.bool)
    gold = torch.zeros(len(items),len(TYPES),length,length) if labels else None
    unknown = vocab.get('[UNK]',1)
    for i,w in enumerate(items):
        n = len(w['text'])
        ids[i,offset:offset+n] = torch.tensor([vocab.get(ch,unknown) for ch in w['text']])
        attention[i,:n+2*offset] = True
        chars[i,offset:offset+n] = True
        if offset:
            ids[i,0],ids[i,n+1] = vocab['[CLS]'],vocab['[SEP]']
        if labels:
            for e in w['entities']:
                gold[i,TYPES.index(e['type']),e['start']+offset,e['end']-1+offset] = 1
    return ids.to(device),attention.to(device),chars.to(device),gold.to(device) if labels else None


def decode(logits, items, method, threshold):
    offset = 1 if method=='N1' else 0
    all_entities = [[] for _ in items]
    for i,typ,a,b in (logits > threshold).nonzero().cpu().tolist():
        a, b = a-offset, b-offset+1
        text = items[i]['text']
        if not 0 <= a < b <= len(text):
            raise ValueError('解码输出越界；遮罩失效')
        all_entities[i].append(dict(start=a,end=b,type=TYPES[typ],text=text[a:b]))
    return all_entities
