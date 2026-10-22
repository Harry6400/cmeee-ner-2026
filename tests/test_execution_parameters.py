"""CPU and mocked-transport checks for actual external settings;no GPU or paid API."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from execution_parameters import load_execution,make_optimizer,clip_gradients,validate_training_resume
from teacher_api import TeacherAPI,api_presets
ROOT=Path(__file__).resolve().parents[1]
class ExecutionParameterTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path=self.root/'settings.json';self.settings=load_execution();self.path.write_text(json.dumps(self.settings))
 def tearDown(self):self.tmp.cleanup()
 def save(self):self.path.write_text(json.dumps(self.settings))
 def models(self):
  torch.manual_seed(412);a=torch.nn.Linear(3,2);b=copy.deepcopy(a);return a,b
 def update(self,m,opt,t=None):
  x=torch.tensor([[3.,-2.,1.],[1.,2.,4.]]);loss=(m(x)**2).sum();loss.backward()
  norm=clip_gradients(m.parameters(),t) if t else torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
  opt.step();return float(norm)
 def test_cpu_default_update_is_bitwise_legacy_equivalent(self):
  a,b=self.models();oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=make_optimizer([{'params':b.parameters(),'lr':.001}],self.settings['training'])
  na,nb=self.update(a,oa),self.update(b,ob,self.settings['training']);self.assertEqual(na,nb)
  for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
  for x,y in zip(oa.state.values(),ob.state.values()):
   for k in x:self.assertTrue(torch.equal(x[k],y[k]) if isinstance(x[k],torch.Tensor) else x[k]==y[k])
 def test_changed_optimizer_and_clip_reach_actual_update(self):
  a,b=self.models();self.settings['training']['optimizer'].update(weight_decay=.2,betas=[.8,.95],eps=.001);self.settings['training']['gradient_clip_norm']=.05;self.save();t=load_execution(self.path)['training'];oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=make_optimizer([{'params':b.parameters(),'lr':.001}],t)
  self.update(a,oa);self.update(b,ob,t);self.assertLessEqual(sum(float(p.grad.square().sum()) for p in b.parameters())**.5,.050001)
  self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())));self.assertEqual(ob.param_groups[0]['betas'],(.8,.95))
 def parent(self,t):
  a,_=self.models();opt=make_optimizer([{'params':a.parameters(),'lr':.001}],t);return dict(optimizer=opt.state_dict())
 def test_legacy_resume_only_accepts_historical_parameters(self):
  t=self.settings['training'];p=self.parent(t);validate_training_resume(p,t)
  changed=copy.deepcopy(t);changed['gradient_clip_norm']=.5
  with self.assertRaisesRegex(ValueError,'Legacy checkpoint'):validate_training_resume(p,changed)
 def test_new_resume_checks_saved_parameters_and_optimizer(self):
  t=self.settings['training'];p=self.parent(t);p['execution_parameters']=copy.deepcopy(t);validate_training_resume(p,t)
  p['execution_parameters']['gradient_clip_norm']=.5
  with self.assertRaisesRegex(ValueError,'parameters differ'):validate_training_resume(p,t)
  p['execution_parameters']=copy.deepcopy(t);p['optimizer']['param_groups'][0]['weight_decay']=.5
  with self.assertRaisesRegex(ValueError,'optimizer settings differ'):validate_training_resume(p,t)
 def test_missing_invalid_and_incomplete_config_fail_explicitly(self):
  with self.assertRaisesRegex(ValueError,'Cannot read'):load_execution(self.root/'absent.json')
  self.path.write_text('{bad')
  with self.assertRaises(ValueError):load_execution(self.path)
  self.path.write_text('{}')
  with self.assertRaises(ValueError):TeacherAPI(self.root,self.path)
  self.assertFalse((self.root/'output').exists())
 def test_secrets_or_unknown_execution_fields_are_not_persisted(self):
  self.settings['api']['api_key']='synthetic-file-secret';self.save()
  with self.assertRaises(ValueError):TeacherAPI(self.root,self.path)
  self.assertFalse((self.root/'output').exists())
  self.settings['api'].pop('api_key');self.settings['api']['model_presets'][0]['api_key']='synthetic-file-secret';self.save()
  with self.assertRaises(ValueError):load_execution(self.path)
 def test_invalid_training_values_are_rejected(self):
  for change in [dict(gradient_clip_norm=0),dict(gradient_clip_norm=float('nan')),dict(optimizer=dict(name='AdamW',betas=[1,.99],eps=1e-8,weight_decay=.01,amsgrad=False))]:
   with self.subTest(change=change):
    d=copy.deepcopy(self.settings);d['training'].update(change);self.path.write_text(json.dumps(d))
    with self.assertRaises(ValueError):load_execution(self.path)
 def test_unsupported_stream_thinking_and_limits_rejected(self):
  for k,v in [('stream',True),('thinking','enabled'),('max_output_tokens',0),('temperature',True),('default_model','absent')]:
   with self.subTest(k=k):
    d=copy.deepcopy(self.settings);d['api'][k]=v;self.path.write_text(json.dumps(d))
    with self.assertRaises(ValueError):load_execution(self.path)
 def reply(self,completion=10):return 200,json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content='{"entities":[]}'))],usage=dict(prompt_tokens=30,completion_tokens=completion))).encode()
 def test_altered_api_settings_control_payload_estimate_and_snapshot(self):
  self.settings['api'].update(default_model='deepseek-v4-pro',temperature=.25,max_output_tokens=512);self.save();api=TeacherAPI(self.root,self.path);preset=api_presets(self.path);self.assertEqual(preset['default_model'],'deepseek-v4-pro')
  payload=dict(request_id='cfg-test',text='发热',prompt_mode='L0',config=dict(provider='deepseek',api_key='synthetic-execution-secret-key'))
  with patch.object(api,'_send',return_value=self.reply()) as send:r=api.extract(payload)
  cfg,body,_=send.call_args.args;self.assertEqual(body['model'],'deepseek-v4-pro');self.assertEqual(body['temperature'],.25);self.assertEqual(body['max_tokens'],512);self.assertFalse(body['stream']);self.assertEqual(body['thinking'],{'type':'disabled'})
  self.assertEqual(cfg['max_output_tokens'],512);self.assertNotIn('reservation_cny',cfg);self.assertEqual(r['cost']['state'],'estimated')
  record=json.loads((self.root/'output/teacher-api/requests/cfg-test/execution-parameters.json').read_text());self.assertEqual(record,self.settings)
  self.assertNotIn('synthetic-execution-secret-key',json.dumps(record))
 def test_api_usage_above_configured_output_cap_remains_unknown(self):
  self.settings['api']['max_output_tokens']=8;self.save();api=TeacherAPI(self.root,self.path)
  with patch.object(api,'_send',return_value=self.reply(completion=10)):r=api.extract(dict(request_id='over-cap',text='发热',config=dict(api_key='synthetic-execution-secret-key')))
  self.assertEqual(r['cost']['state'],'unknown');self.assertIsNone(r['cost']['estimated_cny'])
 def test_default_live_api_contract_and_frozen_prompt_unchanged(self):
  api=TeacherAPI(self.root,self.path);cfg=api.check(dict(provider='deepseek',api_key='synthetic-execution-key'))['config'];self.assertEqual(cfg['model'],'deepseek-flash');self.assertNotIn('reservation_cny',cfg);self.assertEqual(cfg['max_output_tokens'],4096);self.assertEqual(cfg['temperature'],0)
  frozen=json.loads((ROOT/'configs/stage4-frozen-protocol.json').read_text())['api'];self.assertEqual(api._messages('发热','L0')[0]['content'],frozen['system'])
 def test_development_loop_uses_configured_micro_batches_and_logs_effective_values(self):
  import train_development as trainer
  calls=[];original=trainer.ROOT;root=self.root/'study';(root/'configs').mkdir(parents=True);(root/'data/processed').mkdir(parents=True)
  cfg=json.loads((ROOT/'configs/stage3.json').read_text());cfg.update(micro_batch=8,accumulation=4,n0_max_epochs=1)
  (root/'configs/stage3.json').write_text(json.dumps(cfg));(root/'configs/execution-parameters.json').write_text(json.dumps(self.settings))
  class Tiny(torch.nn.Module):
   def __init__(self,*args):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
   def to(self,*args,**kwargs):return self
   def forward(self,ids,attn,chars):calls.append(len(ids));return self.weight*ids
  row=dict(id='synthetic',text='发热',entities=[])
  for split in ['train','dev']:(root/f'data/processed/{split}.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
  metrics=dict(micro=dict(f1=.5,precision=.5,recall=.5),nested_recall=0,character_type_micro=dict(f1=.5),by_type={t:dict(f1=0) for t in trainer.TYPES})
  def collate(batch,*args):return torch.ones(len(batch)),None,None,torch.zeros(len(batch))
  out=root/'new-run'
  with patch.object(trainer,'ROOT',root),patch.object(trainer,'cuda_gate',return_value={}),patch.object(trainer,'configure'),patch.object(trainer,'read_rows',return_value=[row]),patch.object(trainer,'windows',return_value=[row]*32),patch.object(trainer,'make_vocab',return_value={}),patch.object(trainer,'Extractor',Tiny),patch.object(trainer,'collate',side_effect=collate),patch.object(trainer,'loss_fn',side_effect=lambda x,y:(x-y).square().mean()),patch.object(trainer,'evaluate',return_value=([row],metrics,dict(loss=1.,seconds=.01))),patch.object(trainer.signal,'signal'),patch.object(trainer.torch.cuda,'reset_peak_memory_stats'),patch.object(trainer.torch.cuda,'synchronize'),patch.object(trainer.torch.cuda,'max_memory_allocated',return_value=0),patch.object(trainer.torch.cuda,'get_device_name',return_value='synthetic CPU'),patch.object(trainer.torch.cuda,'get_rng_state_all',return_value=[]),patch.object(sys,'argv',['train_development.py','--method','N0','--output',str(out)]):
   trainer.main()
  self.assertEqual(calls,[8,8,8,8]);m=json.loads((out/'metadata.json').read_text());self.assertEqual((m['micro_batch'],m['accumulation'],m['batch']),(8,4,32));self.assertEqual(json.loads((out/'summary.json').read_text())['global_step'],1)
  checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False);self.assertEqual(checkpoint['execution_parameters'],self.settings['training'])
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  import math
  events=EventAccumulator(str(out/'tensorboard')).Reload()
  for tag in ['train/loss','train/lr','train/grad_norm','train/optimizer_step','train/throughput_samples_per_sec','train/epoch_time_sec','system/gpu_memory_peak_mb','val/f1','val/loss','val/character_f1']:
   self.assertIn(tag,events.Tags()['scalars']);self.assertTrue(all(math.isfinite(e.value) for e in events.Scalars(tag)))
  self.assertTrue(any('config/execution_parameters' in tag for tag in events.Tags()['tensors']))
 def test_startup_snapshot_is_stable_after_file_edit(self):
  api=TeacherAPI(self.root,self.path);self.settings['api']['max_output_tokens']=128;self.save();self.assertEqual(api.check(dict(provider='deepseek',api_key='synthetic-execution-key'))['config']['max_output_tokens'],4096);self.assertEqual(TeacherAPI(self.root,self.path).check(dict(provider='deepseek',api_key='synthetic-execution-key'))['config']['max_output_tokens'],128)
if __name__=='__main__':unittest.main()
