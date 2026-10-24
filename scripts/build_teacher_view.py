"""Build the teacher-facing public summary from real experiment artifacts; never read credentials."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def build(root=ROOT):
 r=Path(root);old=r/'ui/project-data.json';prior=json.loads(old.read_text()) if old.exists() else {}
 profile=r/'private/stage6/delivery-profile.json'
 prof=json.loads(profile.read_text()) if profile.exists() else {'people':prior.get('people',[]),'model_share_url':prior.get('links',{}).get('model'),'video_share_url':prior.get('links',{}).get('video')}
 agg=json.loads((r/'evidence/stage4/final-aggregation.json').read_text());common=[];full=[]
 for split,target,methods in [('common500',common,['D0','N0','N1','L0','L3']),('holdout5000',full,['D0','N0','N1'])]:
  for method in methods:
   key=method+'-seed20261019' if method in ['N0','N1'] else method;v=agg['results'][key+'/'+split];m=(v['main_metrics'] or {}).get('micro',{})
   target.append(dict(method=method,valid=v['valid'],total=v['samples_expected'],precision=m.get('precision'),recall=m.get('recall'),f1=m.get('f1'),statuses=v['statuses']))
 multi=[]
 for method in ['N0','N1']:
  for split,d in agg['multi_seed'][method].items():
   v=d['f1'];multi.append(dict(method=method,split=split,mean=v['mean'],sd=v['sample_sd'],seeds=v['seeds'],values=v['values'],ddof=1))
 curves=[];learning=[]
 for name in ['N1-lr-1e-05','N1-lr-3e-05','N1-lr-5e-05','N1-rope-off']:
  summary=json.loads((r/'runs/stage3-20261019'/name/'summary.json').read_text());curves.append(dict(method=name,records=[dict(epoch=x['epoch'],train_loss=x['train_loss'],dev_f1=x['val']['micro']['f1']) for x in summary['records']]))
  if name!='N1-rope-off':learning.append(dict(lr=name.split('-lr-')[1],dev_f1=summary['best_f1']))
 obs=json.loads((r/'evidence/prestage6-review/n1-error-observations.json').read_text())['counts']
 rope=json.loads((r/'report-source/source-data/rope-data.json').read_text());n1=agg['results']['N1-seed20261019/common500']['main_metrics']
 cases=json.loads((r/'ui/demo-cases.json').read_text())['cases'];nested=next(x['case']['input'] for x in cases if x['case']['case_id']=='F04')
 previous=[]
 p=r/'evidence/stage6/unzip-reproduction-result.json'
 if p.exists():
  v=json.loads(p.read_text())['fresh_CPU'];previous.append(dict(label='阶段六初版：候选包解压后CPU重评',at=v['started_at'],seconds=v['seconds'],status=v['status'],source='evidence/stage6/unzip-reproduction-result.json'))
 else:previous=prior.get('previous_reproductions',[])
 sources=['evidence/stage4/final-aggregation.json','runs/stage3-20261019/*/summary.json','report-source/source-data/rope-data.json','evidence/prestage6-review/n1-error-observations.json','runs/stage5-20261020/summary.json']
 obj=dict(schema_version=1,title='中文医学实体抽取',generated_at=datetime.now(timezone.utc).isoformat(),people=prof['people'],teacher_simulation=True,disclosure='示范角色的责任映射与AI制作范围见报告附录二；排版姓名是示范署名。',windows=prior.get('windows','未验收'),samples=[dict(id='normal',label='普通文本',text='出现发热与咳嗽，使用布洛芬。'),dict(id='nested',label='嵌套难例',text=nested),dict(id='emoji',label='特殊字符反例',text='😀 出现发热与咳嗽，使用布洛芬。')],experiments=dict(primary_seed=20261019,common500=common,holdout5000=full,multi_seed=multi,learning_rates=learning,rope=[dict(condition='固定3轮（1302次更新）',on=rope['fixed3_on']/100,off=rope['fixed3_off']/100),dict(condition='相同早停规则各自最佳',on=rope['best_on']/100,off=rope['best_off']/100)],curves=curves,error_groups=[dict(label='同边界错类型',count=obs['same_span_wrong_type']),dict(label='同类型重叠边界',count=obs['same_type_overlapping_boundary']),dict(label='其余遗漏',count=obs['other_missed_reference'])],error_counts=dict(gold=obs['gold'],tp=obs['tp'],fp=obs['fp'],fn=obs['gold']-obs['tp']),by_type=n1['by_type'],nested_recall=n1['nested_recall'],oov_recall=n1['unseen_surface_strict']['recall'],sources=sources,notes=['共同500是完整5000的固定子集，不能当作两份独立测试。','主对比固定20261019，另外两个种子全部报告；均值±样本标准差，ddof=1。','L0/L3原各500次尝试，非法输出和传输失败保留，因此完整主F1为N/A。','combined补丁开发F1从42.3103%降至42.1112%，未替代默认。','错误归组是输出观察，不是模型内部机制的因果证明。']),previous_reproductions=previous,links=dict(data='https://tianchi.aliyun.com/dataset/95414',model=prof.get('model_share_url'),video=prof.get('video_share_url')),submission=dict(status='本地候选；待用户自录视频、两项夸克链接及最终验收',source_zip='中文医学实体抽取_本地候选.zip',final_zip='中文医学实体抽取.zip',video='用户自行真实录屏≤120秒；旧静音试录不再作为交付依据'))
 # Presentation-only supplement: read the finalized report source; do not alter experiment artifacts.
 training=r/'report-source/source-data/training-records.json'
 completion=r/'report-source/source-data/gpu-completion.json'
 report=r/'report-source/source-data/report-data.json'
 if training.is_file() and completion.is_file() and report.is_file():
  curves=json.loads(training.read_text());verify=json.loads(completion.read_text())['verification'];report_data=json.loads(report.read_text())
  assert curves['unique_runs']==10 and len(curves['runs'])==10 and verify['status']=='PASS'
  assert len({row['run'] for row in curves['runs']})==10 and all(len(row['records'])==row['epochs'] for row in curves['runs'])
  exp=obj['experiments'];exp.update(training_curves=curves['runs'],learning_rate_rope_grid=verify['learning_rate_rope_grid'],paired_seeds=verify['paired_seeds'],paired_statistics=verify['paired_statistics'],dev_samples=verify['dev_samples'],supplement_scope='retrospective internal-dev supplement;formal evaluation unchanged',threshold=[dict(method=row['method'],threshold=row['threshold'],**row['metrics']['micro']) for row in report_data['threshold']])
  exp['sources']+=['report-source/source-data/training-records.json','report-source/source-data/gpu-completion.json','report-source/source-data/report-data.json']
  exp['notes'].append('2026-10-21开发补充研究：10组独立任务、3个学习率和3个配对种子；不替换正式测试。')
 old.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');return obj
if __name__=='__main__':
 d=build();print('Built actual public experiment summaries:',len(d['experiments']['common500']),'common routes;',len(d['experiments']['curves']),'real curves;no API or GPU')
