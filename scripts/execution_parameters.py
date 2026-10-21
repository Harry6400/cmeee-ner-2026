"""Read explicit execution settings;no credentials,network or runtime side effects."""
from pathlib import Path
import copy,json,math
from urllib.parse import urlsplit
DEFAULT_PATH=Path(__file__).resolve().parents[1]/'configs/execution-parameters.json'
# Historical compatibility check only;fresh runs always consume the external file.
LEGACY_TRAINING_PARAMETERS=dict(optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,amsgrad=False),gradient_clip_norm=1.0)
def number(v,lo,hi,label,positive=False):
 if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi or (positive and v<=0):raise ValueError('Invalid execution setting: '+label)
 return v

def load_execution(path=None):
 path=Path(path) if path is not None else DEFAULT_PATH
 try:d=json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError) as e:raise ValueError('Cannot read execution configuration: '+str(path)) from e
 try:
  if d['schema_version']!=1:raise ValueError('Unsupported execution configuration version')
  for value,keys in [(d,{'schema_version','training','api'}),(d['training'],{'optimizer','gradient_clip_norm'}),(d['training']['optimizer'],{'name','betas','eps','weight_decay','amsgrad'}),(d['api'],{'base_url','default_model','model_presets','temperature','max_output_tokens','thinking','stream','json_mode','price_source','price_checked_at'})]:
   if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Unknown or missing execution configuration fields')
  t=d['training'];o=t['optimizer']
  if o['name']!='AdamW' or type(o['amsgrad']) is not bool:raise ValueError('Unsupported optimizer')
  if not isinstance(o['betas'],list) or len(o['betas'])!=2:raise ValueError('Invalid AdamW betas')
  for v in o['betas']:
   number(v,0,1,'betas')
   if v>=1:raise ValueError('Invalid AdamW beta')
  number(o['eps'],0,1,'eps',True);number(o['weight_decay'],0,100,'weight_decay');number(t['gradient_clip_norm'],0,1e6,'gradient_clip_norm',True)
  a=d['api'];base=urlsplit(a['base_url'])
  if base.scheme!='https' or base.hostname!='api.deepseek.com' or base.username or base.password or base.query or base.fragment or base.port not in (None,443):raise ValueError('Invalid preset service URL')
  number(a['temperature'],0,2,'temperature')
  if type(a['max_output_tokens']) is not int or not 1<=a['max_output_tokens']<=1048576:raise ValueError('Invalid output limit')
  if a['thinking']!='disabled' or a['stream'] is not False:raise ValueError('This extraction reader requires thinking disabled and non-streamed responses')
  if type(a['json_mode']) is not bool:raise ValueError('Invalid json_mode')
  if not isinstance(a['model_presets'],list) or not a['model_presets']:raise ValueError('Missing model presets')
  ids=[]
  for m in a['model_presets']:
   if not isinstance(m,dict) or set(m)!={'id','label','input_price_cny','output_price_cny','context_max_tokens','json_mode'}:raise ValueError('Unknown or missing preset fields')
   for k in ['id','label']:
    if not isinstance(m[k],str) or not m[k].strip() or len(m[k])>200 or any(ord(c)<32 for c in m[k]):raise ValueError('Invalid model identity')
   ids.append(m['id'])
   for k in ['input_price_cny','output_price_cny']:
    # Preserve decimal-string pricing for the existing Decimal billing calculation.
    from decimal import Decimal
    v=Decimal(m[k])
    if not v.is_finite() or v<0:raise ValueError('Invalid price')
   if type(m['context_max_tokens']) is not int or not a['max_output_tokens']<=m['context_max_tokens']<=10000000:raise ValueError('Invalid context limit')
   if type(m['json_mode']) is not bool:raise ValueError('Invalid preset JSON mode')
  if len(ids)!=len(set(ids)) or a['default_model'] not in ids:raise ValueError('Invalid default or duplicate model')
  for k in ['price_source','price_checked_at']:
   if not isinstance(a[k],str) or not a[k].strip():raise ValueError('Missing price provenance')
 except (KeyError,TypeError,AssertionError,ArithmeticError) as e:raise ValueError('Malformed execution configuration: '+str(path)) from e
 return copy.deepcopy(d)

def make_optimizer(groups,training):
 import torch
 o=training['optimizer'];return torch.optim.AdamW(groups,betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'],amsgrad=o['amsgrad'])
def clip_gradients(parameters,training):
 import torch
 return torch.nn.utils.clip_grad_norm_(parameters,training['gradient_clip_norm'],error_if_nonfinite=True)
def validate_training_resume(parent,training):
 recorded=parent.get('execution_parameters')
 if recorded is None:
  if training!=LEGACY_TRAINING_PARAMETERS:raise ValueError('Legacy checkpoint permits historical execution defaults only')
 else:
  if recorded!=training:raise ValueError('Checkpoint execution parameters differ')
 o=training['optimizer']
 for g in parent['optimizer']['param_groups']:
  if list(g['betas'])!=o['betas'] or any(g[k]!=o[k] for k in ['eps','weight_decay','amsgrad']):raise ValueError('Checkpoint optimizer settings differ')
