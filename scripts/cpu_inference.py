"""Portable FP32 inference for the fixed development-selected seed. No SSH, ledger or API."""
import hashlib,json,time
from pathlib import Path
from collections import defaultdict
import torch
from ner_core import ROOT,windows,merge,validate
from ner_model import Extractor,configure,collate,decode

class CPUModels:
    def __init__(self, root=ROOT, config_path=None):
        self.root=Path(root);self.config_path=Path(config_path) if config_path else self.root/'configs/review-runtime.json'
        self.runtime=json.loads(self.config_path.read_text());self.models={};self.loads={}
        self.manifest=json.loads((self.root/self.runtime['model_manifest']).read_text())
        configure(self.runtime['selected_model_seed'],self.runtime['cpu_threads'])
    def load(self,method):
        if method in self.models:return self.models[method]
        if method not in ('N0','N1'):raise ValueError('unsupported neural method')
        start=time.perf_counter();item=next(x for x in self.manifest['items'] if x['method']==method);p=self.root/'models/trained'/item['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('inference weights differ from manifest')
        state=torch.load(p,map_location='cpu',weights_only=True)
        if state['method']!=method or state['config']['seed']!=self.runtime['selected_model_seed'] or state['provenance']['source_checkpoint_sha256']!=item['source_checkpoint_sha256']:raise ValueError('wrong selected checkpoint')
        model=Extractor(method,state['config'],state['vocab'],state['bert_config']);model.load_state_dict(state['model'],strict=True);model.eval()
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('CPU FP32 required')
        self.models[method]=(model,state);self.loads[method]=time.perf_counter()-start
        return model,state
    def predict_many(self,inputs,method,progress=None):
        if any(set(r)!= {'id','text'} or not isinstance(r['id'],str) or not isinstance(r['text'],str) or not r['text'] for r in inputs):raise ValueError('prediction accepts nonempty id/text only')
        if len({r['id'] for r in inputs})!=len(inputs):raise ValueError('duplicate input ids')
        model,state=self.load(method);cfg=state['config'];start=time.perf_counter();parts=defaultdict(list)
        ws=[w for r in inputs for w in windows(dict(r,entities=[]),cfg['window_chars'],cfg['overlap_chars'])]
        micro=self.runtime['inference_micro_batch']
        if micro!=cfg['micro_batch']:raise ValueError('frozen micro batch differs')
        with torch.inference_mode():
            for i in range(0,len(ws),micro):
                batch=ws[i:i+micro];ids,attn,mask,_=collate(batch,state['vocab'],method,'cpu',labels=False);logits=model(ids,attn,mask)
                if not torch.isfinite(logits).all():raise ValueError('nonfinite inference')
                for w,ents in zip(batch,decode(logits,batch,method,cfg['decode_threshold'])):parts[w['id']].append((w,ents))
                if progress and (i//micro)%100==0:progress(dict(method=method,completed_windows=min(i+micro,len(ws)),total_windows=len(ws),seconds=time.perf_counter()-start))
        outputs=[merge(dict(r,entities=[]),parts[r['id']]) for r in inputs]
        timing=dict(device='cpu',precision='fp32',method=method,seed=cfg['seed'],samples=len(inputs),windows=len(ws),load_seconds=self.loads[method],inference_seconds=time.perf_counter()-start,model_sha256=next(x['sha256'] for x in self.manifest['items'] if x['method']==method),new_API_calls=0)
        return outputs,timing

def read_jsonl(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l]
