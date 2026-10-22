'use strict';
(() => {
const $ = id => document.getElementById(id);
const names = {dis:'疾病',sym:'临床表现',dru:'药物',equ:'医疗设备',pro:'医疗过程',bod:'身体部位与物质',ite:'医学检验',mic:'微生物',dep:'科室'};
const methods = {D0:'D0 · 词典匹配',N0:'N0 · 字符模型',N1:'N1 · BERT 实体抽取',L0:'API · 零样本',L3:'API · 三样本'};
const state = {view:'live',project:null,apiPresets:[],presetsReady:false,configChecked:false,configVersion:0,modelCheck:'unverified',connectionBusy:false,live:{version:0,busy:false,result:null},compare:{version:0,busy:false,result:null},cases:{version:0,data:null,result:null},job:null,poll:null,apiBusy:false,messages:{},experimentTab:'formal',compareOnlyDiff:true,keyboard:false};
const el = (tag,text,cls) => {const e=document.createElement(tag);if(text!==undefined&&text!==null)e.textContent=String(text);if(cls)e.className=cls;return e;};
const show = (id, visible=true) => {$(id).hidden=!visible;};
const fmt = value => value===null||value===undefined||!Number.isFinite(Number(value))?'N/A':`${(Number(value)*100).toFixed(2)}%`;
const seconds = value => Number.isFinite(Number(value))?`${Number(value).toFixed(2)} 秒`:'—';
const safeError = err => userError(err instanceof Error ? err.message : String(err || '操作失败'));
const isApi = method => method==='L0'||method==='L3';
function announce(message,view=state.view){state.messages[view]=message;if(view===state.view)$('footer-status').textContent=message;}
function badge(text,kind='neutral'){return el('span',text,`badge ${kind}`);}
function sourceLabel(source){if(!source)return '来源未标注';if(source==='teacher_live_api')return '本次现场 API 调用';if(String(source).includes('frozen')||String(source).includes('replay'))return '冻结历史预测回放';if(String(source).includes('cpu')||String(source).includes('live'))return '本次本地 CPU 推理';return String(source);}
function formatDate(value){if(!value)return '时间未记录';const date=new Date(value);return Number.isNaN(+date)?String(value):date.toLocaleString('zh-CN',{hour12:false});}
async function request(url,body){const response=await fetch(url,{method:body===undefined?'GET':'POST',headers:body===undefined?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store'});let data;try{data=await response.json();}catch{throw Error(`服务返回了无法解析的响应（HTTP ${response.status}）。`);}if(!response.ok)throw Object.assign(Error(data.error||data.message||`请求失败（HTTP ${response.status}）。`),{error_code:data.error_code,stage:data.stage,request_sent:data.request_sent,http_status:data.http_status});return data;}
const numeric = label => ({label,align:'right'}), centered = label => ({label,align:'center'});
function table(headers,rows,cls=''){
  const columns=headers.map(header=>typeof header==='string'?{label:header,align:'left'}:header),wrap=el('div',null,'table-wrap'),t=el('table',null,`data-table ${cls}`),head=el('thead'),tr=el('tr');
  columns.forEach(column=>{const th=el('th',column.label,`align-${column.align}`);th.setAttribute('scope','col');tr.append(th);});head.append(tr);t.append(head);
  const body=el('tbody');rows.forEach(row=>{const r=el('tr');row.forEach((v,i)=>{const td=el('td',null,`align-${columns[i]?.align||'left'}`);if(v instanceof Node)td.append(v);else td.textContent=v===null||v===undefined?'—':String(v);r.append(td);});body.append(r);});t.append(body);wrap.append(t);return wrap;
}
function clearLiveActions(){['live-json','live-copy','live-export'].forEach(id=>$(id).disabled=true);show('live-json-details',false);$('live-json-content').textContent='';}
function invalidateLive(message='原文或方法已改变，请重新运行。'){
  state.live.version++;state.live.result=null;clearLiveActions();$('live-result').replaceChildren(empty(message));
  $('live-result-state').textContent='待重新运行';$('live-result-state').className='badge warning';$('live-error').textContent='';
  updateCounts();announce(message,'live');updateRuntime();
}
function invalidateCompare(){state.compare.version++;state.compare.result=null;$('compare-export').disabled=true;$('compare-result').replaceChildren(empty('输入或路线已改变，请重新运行对比。'));$('compare-state').textContent='待重新运行';$('compare-state').className='badge warning';$('compare-error').textContent='';updateCounts();announce('对比输入已更新，等待运行。','compare');}
function empty(text){const box=el('div',null,'empty-state');box.append(el('p',text));return box;}
function updateCounts(){$('live-char-count').textContent=`${Array.from($('live-text').value).length} / 4000 字符`;$('compare-count').textContent=`${Array.from($('compare-text').value).length} / 4000 字符`;}
function validateText(text,id){if(!text.trim()||Array.from(text).length>4000){$(id).setAttribute('aria-invalid','true');throw Error('请输入 1—4000 个 Unicode 字符，不能仅含空白。');}$(id).removeAttribute('aria-invalid');}
function validEntities(text,entities){if(!Array.isArray(entities))return null;const chars=Array.from(text);return entities.filter(e=>Number.isInteger(e.start)&&Number.isInteger(e.end)&&e.start>=0&&e.end>e.start&&e.end<=chars.length).map(e=>({...e,text:e.text||chars.slice(e.start,e.end).join('')}));}
function overlapping(entity,entities){return entities.some(other=>other!==entity&&entity.start<other.end&&other.start<entity.end);}
function renderReading(text,entities,selected,diagnostics=null){
  const box=el('div',null,'reading'),chars=Array.from(text),bounds=new Set([0,chars.length]);
  entities.forEach(e=>{bounds.add(e.start);bounds.add(e.end);});const points=[...bounds].sort((a,b)=>a-b);
  for(let i=0;i<points.length-1;i++){
    const a=points[i],b=points[i+1],active=entities.filter(e=>e.start<=a&&b<=e.end),fragment=chars.slice(a,b).join('');
    if(active.length){const mark=el('mark',fragment,active.length>1?'nested':null);
      mark.title=active.map(e=>`${e.text} · ${names[e.type]||e.type} [${e.start}, ${e.end})`).join('\n');
      const diagnostic=active.map(e=>diagnostics?.get(entityKey(e))).find(d=>d&&d.kind!=='exact');
      if(diagnostic)mark.classList.add('diag-'+diagnostic.kind);
      if(selected&&selected.start<b&&a<selected.end)mark.classList.add('selected');box.append(mark);
    }else box.append(document.createTextNode(fragment));
  }return box;
}
let entityViewId=0;
function entityView(text,rawEntities,{status='valid',raw,error,label='实体结果',diagnostics=null,onlyDifferences=false,onSelect=null}={}){
  const wrap=el('div',null,'entity-view'),entities=validEntities(text,rawEntities);
  if(entities===null){const invalid=el('div',null,'invalid-output');invalid.append(el('strong',status==='transport_failure'||status==='transport_error'?'传输失败，未产生完整预测':status==='http_error'?'API 服务返回错误':'预测无效，不能按空实体处理'));
    if(error)invalid.append(el('p',typeof error==='string'?error:JSON.stringify(error)));wrap.append(invalid);
    if(raw!==undefined&&raw!==null){const detail=el('details');detail.append(el('summary','保留的原始输出'),el('pre',typeof raw==='string'?raw:JSON.stringify(raw,null,2),'json-box'));wrap.append(detail);}return wrap;
  }
  let selected=null,filter='all';const readingHost=el('div'),controls=el('div',null,'entity-controls'),count=el('span',null,'entity-count'),filterWrap=el('div'),filterId=`entity-filter-${++entityViewId}`,filterLabel=el('label','筛选类别'),select=el('select');
  filterLabel.htmlFor=filterId;select.id=filterId;select.setAttribute('aria-label',`${label}：筛选实体类别`);select.append(new Option('全部类别','all'));
  Object.entries(names).forEach(([key,value])=>{if(entities.some(e=>e.type===key))select.append(new Option(value,key));});filterWrap.append(filterLabel,select);controls.append(count,filterWrap);
  const list=el('div',null,'entity-list');list.setAttribute('aria-label',`${label}实体列表`);
  const selectionNote=el('div',null,'entity-detail');
  function draw(){
    const category=filter==='all'?entities:entities.filter(e=>e.type===filter),visible=onlyDifferences&&diagnostics?category.filter(e=>diagnostics.get(entityKey(e))?.kind!=='exact'):category;
    readingHost.replaceChildren(renderReading(text,category,selected,diagnostics));list.replaceChildren();
    count.textContent=visible.length===entities.length?`${entities.length} 个实体`:`显示 ${visible.length} / ${entities.length} 个实体`;
    selectionNote.textContent=selected?`已定位「${selected.text}」 · ${names[selected.type]||selected.type} · Unicode [${selected.start}, ${selected.end})`:'';
    if(!visible.length){list.append(el('p',entities.length?'当前筛选下没有差异实体。':'未抽取到实体；这不等同于模型正确。','help'));return;}
    visible.forEach(entity=>{
      const button=el('button',null,'entity-row');button.type='button';button.setAttribute('data-type',entity.type);button.setAttribute('aria-label',`${entity.text}，${names[entity.type]||entity.type}，位置${entity.start}至${entity.end}`);button.setAttribute('aria-pressed',selected===entity?'true':'false');
      const term=el('span',entity.text,'entity-text'),diagnostic=diagnostics?.get(entityKey(entity));
      if(diagnostic&&diagnostic.kind!=='exact')term.append(el('span',diagnostic.label,`diag-tag ${diagnostic.kind}`));
      if(overlapping(entity,entities))term.append(el('span','嵌套/重叠','nested-label'));
      button.append(el('span',names[entity.type]||entity.type,'entity-type'),term,el('span',`[${entity.start}, ${entity.end})`,'entity-position'));
      button.title=`${entity.type} · Unicode [${entity.start}, ${entity.end})`;
      button.onclick=()=>{selected=entity;draw();onSelect?.(entity);const current=readingHost.querySelector('.selected');if(current)current.scrollIntoView({block:'nearest',behavior:'auto'});const replacement=Array.from(list.querySelectorAll('button')).find(b=>b.getAttribute('aria-pressed')==='true');if(replacement)replacement.focus({preventScroll:true});};list.append(button);
    });
  }
  wrap.selectEntity=entity=>{selected=entities.find(e=>entityKey(e)===entityKey(entity))||entities.find(e=>e.start===entity.start&&e.end===entity.end)||entities.find(e=>e.start<entity.end&&entity.start<e.end)||entity;draw();const current=readingHost.querySelector('.selected');if(current)current.scrollIntoView({block:'nearest',behavior:'auto'});};
  select.onchange=()=>{filter=select.value;selected=null;draw();};draw();wrap.append(readingHost,controls,list,selectionNote);
  const legend=el('div',null,'entity-legend'),ordinary=el('span'),nested=el('span');ordinary.append(el('i'),document.createTextNode('实体覆盖'));nested.append(el('i',null,'nest'),document.createTextNode('嵌套或重叠'));legend.append(ordinary,nested,el('span','点击列表定位'));wrap.append(legend);return wrap;
}

function entityKey(e){return `${e.start}|${e.end}|${e.type}`;}
function diagnose(entities,others,side){
  const otherKeys=new Set(others.map(entityKey)),result=new Map();
  entities.forEach(e=>{
    let kind,label;
    if(otherKeys.has(entityKey(e))){kind='exact';label='一致';}
    else if(others.some(o=>o.start===e.start&&o.end===e.end)){kind='type';label='类型不同';}
    else if(others.some(o=>o.type===e.type&&o.start<e.end&&e.start<o.end)){kind='boundary';label='边界不同';}
    else{kind=side==='reference'?'missing':'extra';label=side==='reference'?'漏检':'多检';}
    result.set(entityKey(e),{kind,label});
  });return result;
}
function updateRuntime(){
  if(!$('runtime-label'))return;let label,kind='local';
  if(state.view==='live'){
    const method=$('live-method').value;
    if(isApi(method)){kind='api';label=state.live.busy?'API · 请求中':state.live.result?.status==='valid'?'API · 本次请求完成':state.configChecked?(state.modelCheck==='verified'?'API · 模型列表已核验':'API · 已本地检查'):'API · 未配置';}
    else label=`本机 CPU · ${method}${state.live.busy?' 运行中':''}`;
  }else label={compare:'同文对比 · 现场推理',errors:'历史回放 · 无新调用',experiments:'已保存实验 · 只读',delivery:'资产与复现验收'}[state.view];
  $('runtime-label').textContent=label;$('runtime-status').setAttribute('data-kind',kind);
}
function reveal(node){
  if(state.keyboard||globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches||!node?.animate)return;
  node.getAnimations?.().forEach(animation=>animation.cancel());
  node.animate([{opacity:.35,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:180,easing:'cubic-bezier(0.23,1,0.32,1)'});
}
function bytes(value){return Number.isFinite(value)?value>=1048576?`${(value/1048576).toFixed(2)} MB`:value>=1024?`${(value/1024).toFixed(1)} KB`:`${value} B`:'体积未记录';}

function userError(message){const text=typeof message==='string'?message:JSON.stringify(message);return /预算|费用|预留|额度|余额|[¥￥]|\d+(?:\.\d+)?\s*元|budget|billing|balance|credit|cost|reservation/i.test(text)?'本次调用受到额度或账户状态保护，请检查本地运行记录及服务商账户。':text;}
function entityDTO(entity){return {start:entity.start,end:entity.end,type:entity.type,text:entity.text};}
function resultDTO(data){const dto={};for(const key of ['request_id','status','text','method','seconds','provider','model','endpoint','source','new_api_calls','started_at','finished_at','http_status','prompt_mode','raw_output','error_code','stage','request_sent'])if(data[key]!==undefined)dto[key]=data[key];dto.prediction=Array.isArray(data.prediction)?data.prediction.map(entityDTO):null;if(data.neural_timing){dto.neural_timing={};for(const key of ['device','precision','windows','load_seconds','inference_seconds'])if(data.neural_timing[key]!==undefined)dto.neural_timing[key]=data.neural_timing[key];}if(data.error)dto.error=userError(data.error);if(data.usage){dto.usage={};for(const key of ['prompt_tokens','completion_tokens','total_tokens','input_tokens','output_tokens'])if(data.usage[key]!==undefined)dto.usage[key]=data.usage[key];}return dto;}
function projectDTO(data){return {title:data.title,generated_at:data.generated_at,people:(data.people||[]).map(person=>({name:person.name,student_id:person.student_id})),samples:data.samples||[],assets:data.assets||{},experiments:data.experiments||{},artifacts:(data.artifacts||[]).map(item=>({id:item.id,label:item.label,available:item.available,url:item.url,status:item.status,file_name:item.file_name,file_size:item.file_size,modified_at:item.modified_at})),previous_reproductions:data.previous_reproductions||[],links:data.links||{}};}
function summaryDTO(data){const result=data.result||data,output={};for(const key of ['status','split','samples','started_at','finished_at','seconds'])if(data[key]!==undefined)output[key]=data[key];output.results={};for(const [method,row] of Object.entries(result.results||{})){const metrics=row.metrics||row;output.results[method]={samples:metrics.samples,micro:metrics.micro,frozen_metric_equal:row.frozen_metric_equal};}return output;}
function apiConfig(){const provider=$('api-provider').value,deep=provider==='deepseek',config={provider,base_url:$('api-base').value.trim(),model:deep?$('api-model').value:$('api-custom-model').value.trim(),api_key:$('api-key').value,json_mode:$('api-json').checked};if(!deep)Object.assign(config,{input_price_cny:Number($('api-input-price').value),output_price_cny:Number($('api-output-price').value),context_max_tokens:Number($('api-context').value),price_source:$('api-price-source').value.trim()});return config;}
function requireApiConfig(){const config=apiConfig();if(!config.api_key.trim())throw Error('请先在右上角“API 设置”填写密钥并检查配置。');if(!state.configChecked)throw Error('API 配置尚未通过本地检查，请在“API 设置”中检查并应用。');return config;}
function requestId(){return globalThis.crypto?.randomUUID?crypto.randomUUID():`teacher-${Date.now()}-${Math.random().toString(16).slice(2)}`;}
async function extract(text,method){if(!isApi(method)){const d=await request('/api/extract',{text,method});return resultDTO({...d,text,method,source:d.source||'teacher_live_cpu',status:d.status||'valid'});}const config=requireApiConfig();if(state.apiBusy)throw Error('已有现场 API 请求正在运行，请等待它完成。');if(method==='L3'&&state.project?.assets?.L3?.ready===false)throw Error('三样本所需官方训练数据尚未准备，请先准备资产。');state.apiBusy=true;try{const d=await request('/api/extract-api',{request_id:requestId(),text,prompt_mode:method,config});return resultDTO({...d,text,method});}finally{state.apiBusy=false;}}
function liveMetadata(data){
  const wrap=el('div'),meta=el('div',null,'result-meta');meta.append(el('span',methods[data.method]||data.method),el('span',`· ${sourceLabel(data.source)}`),el('span',`· 本次总耗时 ${seconds(data.seconds)}`));
  if(isApi(data.method)){if(data.model)meta.append(el('span',`· ${data.model}`));const usage=data.usage;if(usage)meta.append(el('span',`输入 ${usage.prompt_tokens??usage.input_tokens??'未知'} / 输出 ${usage.completion_tokens??usage.output_tokens??'未知'} token`));}
  wrap.append(meta);if(data.neural_timing){const detail=el('details',null,'timing-note'),t=data.neural_timing;detail.append(el('summary','推理计时详情'),el('p',`本次模型推理 ${seconds(t.inference_seconds)} · ${t.windows} 个窗口。模型在本服务首次加载 ${seconds(t.load_seconds)}，此项不是每次重复加载；现场总耗时不等于实验平均延迟。`));wrap.append(detail);}return wrap;
}
function setLiveBusy(busy,method){state.live.busy=busy;$('live-run').disabled=busy;$('live-result').setAttribute('aria-busy',String(busy));show('live-progress',busy);updateRuntime();if(busy){$('live-progress').textContent=`正在运行 ${methods[method]}${isApi(method)?' · 等待服务响应':' · 本机 CPU 加载模型并推理'}，0.0 秒`;const began=performance.now();const timer=setInterval(()=>{if(!state.live.busy)return clearInterval(timer);$('live-progress').textContent=`正在运行 ${methods[method]}${isApi(method)?' · 等待服务响应':' · 本机 CPU 推理'}，${((performance.now()-began)/1000).toFixed(1)} 秒`;},250);}}
async function runLive(){if(state.live.busy)return;const text=$('live-text').value,method=$('live-method').value;try{validateText(text,'live-text');if(isApi(method))requireApiConfig();}catch(err){$('live-error').textContent=safeError(err);if(isApi(method)&&!state.configChecked)openApi();return;}invalidateLive('正在运行，请等待本次抽取结果…');const version=state.live.version;setLiveBusy(true,method);$('live-result-state').textContent='运行中';$('live-result-state').className='badge';try{const data=await extract(text,method);if(version!==state.live.version)return;state.live.result={...data,export_schema:'teacher-entity-result-v1',selected_route:method,text,exported_from:'实时抽取',obtained_at:new Date().toISOString()};$('live-result').replaceChildren(liveMetadata(data),entityView(text,data.prediction,{status:data.status,raw:data.raw_output??data.raw_response,error:data.error,label:methods[method]}));$('live-result-state').textContent=data.status==='valid'?'已完成':'保留失败';$('live-result-state').className=data.status==='valid'?'badge':'badge failure';$('live-json-content').textContent=JSON.stringify(state.live.result,null,2);['live-json','live-copy','live-export'].forEach(id=>$(id).disabled=false);announce(data.status==='valid'?'本次抽取完成，可定位或导出实体。':'本次输出未通过校验，失败原文已保留。','live');updateRuntime();}catch(err){if(version===state.live.version){$('live-error').textContent=safeError(err);$('live-result').replaceChildren(empty('运行未完成，请检查错误后重试。'));$('live-result-state').textContent='运行失败';$('live-result-state').className='badge failure';}}finally{setLiveBusy(false,method);}}
function updateMethodNotice(){
  const method=$('live-method').value;show('live-api-note',isApi(method));
  $('live-api-note').textContent=state.configChecked?(state.modelCheck==='verified'?'模型列表检查通过。开始抽取会发送当前文本，输出以本次结果为准。':'本地检查通过。开始抽取会将当前文本发送至所选服务。'):'尚未配置 API。请先打开右上角 API 设置；本地方法可直接运行。';
  const asset=state.project?.assets?.[method];$('live-asset-note').textContent=asset?.ready===false?asset.message||'所需资产尚未准备。':'';show('live-asset-note',asset?.ready===false);updateRuntime();
}
async function runCompare(){if(state.compare.busy)return;const text=$('compare-text').value,selected=Array.from(document.querySelectorAll('.compare-method:checked')).map(e=>e.value);try{validateText(text,'compare-text');if(!selected.length)throw Error('请至少选择一种方法。');if(selected.some(isApi))requireApiConfig();}catch(err){$('compare-error').textContent=safeError(err);if(selected.some(isApi)&&!state.configChecked)openApi();return;}invalidateCompare();$('compare-result').replaceChildren(empty('正在运行所选方法，请等待本次对比结果…'));const version=state.compare.version;state.compare.busy=true;announce('正在运行所选方法，请等待本次对比结果。','compare');$('compare-run').disabled=true;$('compare-result').setAttribute('aria-busy','true');show('compare-progress');$('compare-state').textContent='运行中';$('compare-state').className='badge';const results=[];try{for(let i=0;i<selected.length;i++){if(version!==state.compare.version)break;const method=selected[i];$('compare-progress').textContent=`${i+1} / ${selected.length} · 正在运行 ${methods[method]}${isApi(method)?'（将外发当前文本）':'（本机 CPU）'}`;try{results.push(await extract(text,method));}catch(err){results.push({text,method,prediction:null,status:'request_failed',error:safeError(err),source:isApi(method)?'teacher_live_api':'teacher_live_cpu'});}if(version!==state.compare.version)break;renderComparison(text,results,selected);}
if(version!==state.compare.version)return;state.compare.result={export_schema:'teacher-method-comparison-v1',text,selected_routes:selected,source:'本次现场同文本方法对比',no_reference:true,results,obtained_at:new Date().toISOString()};$('compare-export').disabled=false;const failed=results.filter(r=>r.status!=='valid').length;$('compare-state').textContent=failed?`已完成 · ${failed} 项失败`:'已完成';$('compare-state').className=failed?'badge warning':'badge';announce('方法对比完成；✓ 仅代表检出，不代表正确。','compare');}finally{state.compare.busy=false;$('compare-run').disabled=false;$('compare-result').setAttribute('aria-busy','false');show('compare-progress',false);}}
function renderComparison(text,results,selected){
  const host=$('compare-result');host.replaceChildren();const union=new Map();
  results.forEach(r=>(r.prediction||[]).forEach(entity=>{if(!union.has(entityKey(entity)))union.set(entityKey(entity),entity);}));
  const entities=[...union.values()].sort((a,b)=>a.start-b.start||a.end-b.end||a.type.localeCompare(b.type));
  const complete=selected.every(method=>results.some(r=>r.method===method&&r.status==='valid'));
  const present=(entity,method)=>results.find(r=>r.method===method)?.prediction?.some(e=>entityKey(e)===entityKey(entity));
  const common=complete?entities.filter(e=>selected.every(method=>present(e,method))):[];
  const summary=el('div',null,'comparison-summary');
  if(complete)summary.append(el('strong',`共同 ${common.length} 项`),el('span',`分歧 ${entities.length-common.length} 项`));
  results.forEach(r=>summary.append(el('span',`${r.method} · ${r.status==='valid'?`${r.prediction?.length||0} 实体 / ${seconds(r.seconds)}`:'失败，保留原始输出'}`)));host.append(summary);
  const tools=el('div',null,'comparison-tools'),filterLabel=el('label',null,'check-label'),filter=el('input');filter.type='checkbox';filter.checked=state.compareOnlyDiff;filter.setAttribute('aria-label','只看方法分歧');filterLabel.append(filter,document.createTextNode('只看分歧'));tools.append(filterLabel,el('span','✓ 仅代表检出；无参考标注，不计算准确率','muted'));host.append(tools);
  const preview=el('div',null,'comparison-preview'),body=el('div');host.append(preview,body);
  function draw(selectedEntity=null){
    const commonKeys=new Set(common.map(entityKey)),visible=state.compareOnlyDiff&&complete?entities.filter(e=>!commonKeys.has(entityKey(e))):entities;
    preview.replaceChildren(renderReading(text,entities,selectedEntity));body.replaceChildren();
    const rows=visible.map(entity=>{
      const identity=el('button',null,'matrix-term');identity.type='button';identity.setAttribute('aria-label',`定位 ${entity.text} ${entity.start}至${entity.end}`);
      identity.append(el('strong',entity.text),el('span',names[entity.type]||entity.type));
      const peers=entities.filter(e=>entityKey(e)!==entityKey(entity)&&e.start<entity.end&&entity.start<e.end);
      if(peers.some(e=>e.start===entity.start&&e.end===entity.end))identity.append(el('span','类型差异','diag-tag'));
      else if(peers.some(e=>e.type===entity.type))identity.append(el('span','边界差异','diag-tag'));
      identity.onclick=()=>{preview.replaceChildren(renderReading(text,entities,entity));};
      return [identity,`[${entity.start}, ${entity.end})`,...selected.map(method=>{const r=results.find(r=>r.method===method),v=el('span',!r?'待运行':r.status!=='valid'?'失败':present(entity,method)?'✓':'—',r?.status==='valid'&&present(entity,method)?'check':'absent');v.title=v.textContent==='✓'?'该方法检出此精确跨度与类型':'该方法没有有效检出';return v;})];
    });
    if(rows.length)body.append(table(['实体 / 类别','Unicode 位置',...selected],rows));
    else body.append(empty(entities.length?'所选方法在当前文本上没有分歧。':'没有有效实体，请查看各路线状态及原始输出。'));
  }
  filter.onchange=()=>{state.compareOnlyDiff=filter.checked;draw();};draw();
  const detail=el('details');detail.append(el('summary','按方法查看完整实体与无效输出'));
  results.forEach(r=>{const section=el('section',null,'experiment-section');section.append(el('h4',methods[r.method]),liveMetadata(r),entityView(text,r.prediction,{status:r.status,raw:r.raw_output??r.raw_response,error:r.error,label:methods[r.method]}));detail.append(section);});host.append(detail);
}
async function loadCases(){try{const cases=await request('/api/cases');const list=Array.isArray(cases)?cases:cases.cases||[];$('case-select').replaceChildren();list.forEach(c=>$('case-select').append(new Option(`${c.case_id} · ${c.category}`,c.case_id)));if(list.length){if(list.some(c=>c.case_id==='F04'))$('case-select').value='F04';loadCase();}else{$('case-status').textContent='当前源码包没有可展示的失败案例。';}}catch(err){$('case-error').textContent=safeError(err);}}
async function loadCase(){const version=++state.cases.version,id=$('case-select').value;$('case-route').disabled=true;state.cases.data=null;state.cases.result=null;$('case-diff').replaceChildren();$('case-export').disabled=true;$('case-results').replaceChildren(empty(`正在加载 ${id}，旧案例已清除。`));$('case-results').setAttribute('aria-busy','true');show('case-analysis',false);show('case-repair',false);show('case-json-details',false);$('case-error').textContent='';$('case-status').textContent=`正在加载 ${id} 的冻结记录；这不是新的模型调用。`;announce(`正在加载 ${id} 历史记录。`,'errors');try{const data=await request(`/api/case?id=${encodeURIComponent(id)}`);if(version!==state.cases.version)return;state.cases.data=data;$('case-route').value=data.case.route;renderCase();}catch(err){if(version===state.cases.version){$('case-error').textContent=safeError(err);$('case-results').replaceChildren();$('case-status').textContent=`${id} 未加载成功，未显示旧预测。`;}}finally{if(version===state.cases.version){$('case-results').setAttribute('aria-busy','false');$('case-route').disabled=false;}}}
function renderCase(){
  const data=state.cases.data;if(!data)return;const c=data.case,route=$('case-route').value,r=data.routes[route];
  state.cases.result=null;$('case-export').disabled=true;$('case-error').textContent='';
  if(!r){$('case-error').textContent=`该案例没有 ${route} 的历史结果。`;$('case-results').replaceChildren();$('case-diff').replaceChildren();return;}
  const reference=c.reference||[],predicted=Array.isArray(r.entities)?r.entities:null,refDiagnostic=predicted?diagnose(reference,predicted,'reference'):null,predDiagnostic=predicted?diagnose(predicted,reference,'prediction'):null;
  const onlyDifferences=$('case-only-diff').checked;let refView,predView;
  function casePanel(title,entities,status,raw,diagnostics,onSelect){
    const panel=el('section',null,'case-panel'),head=el('div',null,'pane-head');head.append(el('h3',title),badge(entities===null?'无有效预测':`${entities.length} 实体`,entities===null?'failure':'neutral'));
    const view=entityView(c.input,entities,{status,raw,label:title,diagnostics,onlyDifferences,onSelect});panel.append(head,view);return {panel,view};
  }
  const left=casePanel('参考标注',reference,'valid',null,refDiagnostic,e=>predView?.selectEntity?.(e));refView=left.view;
  const right=casePanel(`${methods[route]} · 历史预测`,r.entities,r.status,r.wire,predDiagnostic,e=>refView?.selectEntity?.(e));predView=right.view;
  $('case-results').replaceChildren(left.panel,right.panel);
  $('case-status').textContent=`${c.case_id} · ${c.id} · ${sourceLabel(r.source)}`;
  const diff=$('case-diff');diff.replaceChildren(el('h3',route===c.route?c.category:`${route} 与参考标注的差异`));
  diff.append(el('p',route===c.route?c.observed_facts:`当前查看 ${methods[route]}；本案例归因与独立修复针对 ${methods[c.route]}，不混用路线结论。`));
  const counts=el('div',null,'diff-counts');
  if(predicted){const refKeys=new Set(reference.map(entityKey)),predKeys=new Set(predicted.map(entityKey)),tp=[...refKeys].filter(k=>predKeys.has(k)).length;counts.append(badge(`一致 ${tp}`),badge(`漏检 ${refKeys.size-tp}`,'warning'),badge(`多检 ${predKeys.size-tp}`,'failure'));}
  else counts.append(badge('无效输出 · 不能按空实体计数','failure'));
  if(c.target&&route===c.route)counts.append(badge(`目标「${c.target.text}」${c.target_recovered?'已恢复':'仍未恢复'}`,c.target_recovered?'neutral':'warning'));
  diff.append(counts);
  const semantics=el('details');semantics.append(el('summary','差异与计数口径'),el('p','一致要求边界和类别同时匹配。边界/类型差异标签仅作解释，严格计数仍分别计入漏检与多检；不改变冻结评分。','help'));diff.append(semantics);
  $('case-analysis').replaceChildren();const observed=el('section'),cause=el('section');observed.append(el('div','可直接核对的事实','analysis-label'),el('h3',c.category),el('p',c.observed_facts));cause.append(el('div','有依据的归因与范围','analysis-label'),el('h3','结论边界'),el('p',c.attribution));$('case-analysis').append(cause);show('case-analysis');
  const repair=$('case-repair-content');repair.replaceChildren();repair.append(el('p',`补丁作用路线：${methods[c.route]}。当前查看：${methods[route]}。${route!==c.route?'下面的补丁不是当前路线的修复结果。':''}`,'repair-copy'),el('p',c.limitations,'repair-copy'));
  if(c.route==='D0')repair.append(el('p','组合补丁开发 F1：42.3103% → 42.1112%，下降；不替代正式默认方法。','repair-copy'));
  if(c.target)repair.append(el('p',`目标「${c.target.text}」[${c.target.start}, ${c.target.end})：${c.target_recovered?'已恢复（只代表此目标）':'仍未恢复'}。`,'repair-copy'));
  else if(c.case_id==='F03')repair.append(el('p','离线重解析恢复了格式有效性，仍未正确给出参考的 GH血浓度 / GH浓度；未新增 API 调用。','repair-copy'));
  if(c.route==='N1')repair.append(el('p','本轮未修改 N1 权重或解码；该边界错误仍保留。','repair-copy'));
  if(Array.isArray(c.repaired_prediction)){const detail=el('details');detail.append(el('summary',`${c.route} 独立诊断后的实体`),entityView(c.input,c.repaired_prediction,{label:`${c.route}独立诊断`}));repair.append(detail);}
  const evidence=el('details');evidence.append(el('summary','归因依据与原始记录位置'),el('pre',JSON.stringify({source_prediction:c.source_prediction,evidence:c.evidence},null,2),'json-box'));repair.append(evidence);show('case-repair');
  state.cases.result={export_schema:'teacher-case-evidence-v1',selected_case:c.case_id,selected_route:route,text:c.input,reference:c.reference,prediction:r.entities,status:r.status,source:r.source,source_prediction:c.source_prediction,raw_output:r.wire??null,observed_facts:c.observed_facts,attribution:c.attribution,independent_repair:{applies_to:c.route,prediction:c.repaired_prediction,target:c.target,target_recovered:c.target_recovered,limitations:c.limitations},retrospective_known_case:true,new_api_calls:0};
  $('case-json').textContent=JSON.stringify(state.cases.result,null,2);show('case-json-details');$('case-export').disabled=false;announce(`${c.case_id} · ${route} 历史预测，未产生新调用。`,'errors');
}
function exportJson(data,file){if(!data)return;const enriched={...data,exported_at:new Date().toISOString()},url=URL.createObjectURL(new Blob([JSON.stringify(enriched,null,2)],{type:'application/json;charset=utf-8'})),a=el('a');a.href=url;a.download=file;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);announce('已导出当前结果与状态 JSON。');}
async function copyJson(){if(!state.live.result)return;const text=JSON.stringify(state.live.result,null,2);try{await navigator.clipboard.writeText(text);announce('已复制当前结果 JSON。');}catch{$('live-json-details').open=true;show('live-json-details');$('live-error').textContent='浏览器未允许复制；可展开 JSON 手动复制或导出。';}}
function openApi(){$('api-dialog').showModal();$('api-key').focus();}
function closeApi(){globalThis.TeacherSelect?.close();$('api-dialog').close();$('open-api').focus();}
function destinationNote(){return $('api-provider').value==='deepseek'?'官方地址：https://api.deepseek.com':'支持公共 HTTPS 地址、/v1 或完整 /chat/completions 地址。';}
function uncheckApi(){state.configVersion++;state.configChecked=false;state.modelCheck='unverified';$('api-config-dot').classList.remove('ready');$('api-validation').hidden=true;$('api-form-error').textContent='';$('api-endpoint').textContent=destinationNote();updateMethodNotice();}
function updateApiLayout(){const deep=$('api-provider').value==='deepseek';$('api-base').readOnly=deep;$('api-model').disabled=!deep;show('api-official-model-wrap',deep);show('api-base-wrap',!deep);show('api-custom-model-wrap',!deep);$('api-custom-model').disabled=deep;$('api-custom-model').required=!deep;show('api-custom-pricing',!deep);$('api-custom-pricing').disabled=deep;for(const id of ['api-input-price','api-output-price','api-context','api-price-source'])$(id).disabled=deep;document.querySelectorAll('[data-provider]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.provider===$('api-provider').value)));if(!deep)$('api-advanced').open=true;globalThis.TeacherSelect?.refresh();}
function applyProvider(){const deep=$('api-provider').value==='deepseek';$('api-key').value='';$('api-base').value=deep?'https://api.deepseek.com':'';$('api-json').checked=deep;updateApiLayout();uncheckApi();}
function changeAddress(){$('api-key').value='';uncheckApi();}
function focusInvalid(field,message){field.setAttribute('aria-invalid','true');$('api-form-error').textContent=message;if(['api-input-price','api-output-price','api-context','api-price-source','api-json'].includes(field.id))$('api-advanced').open=true;if(field.tagName==='SELECT')globalThis.TeacherSelect?.focus(field);else field.focus();}
function validateApiFields(){const deep=$('api-provider').value==='deepseek';if(!$('api-key').value.trim()){focusInvalid($('api-key'),'请填写 API 密钥。');return false;}if(deep&&!state.presetsReady){$('api-form-error').textContent='本地模型列表尚未就绪，请稍后或刷新页面。';return false;}if(deep&&!state.apiPresets.some(item=>item.id===$('api-model').value)){focusInvalid($('api-model'),'请选择可用模型。');return false;}if(!deep){for(const id of ['api-base','api-custom-model','api-input-price','api-output-price','api-context','api-price-source']){const field=$(id);if(!field.value.trim()||field.validity?.valid===false){focusInvalid(field,'请完整填写兼容接口的地址、模型 ID 和高级设置参数。');return false;}}}return true;}
function applyLocalCheck(data){state.configChecked=true;$('api-config-dot').classList.add('ready');const endpoint=data.endpoint||data.config?.endpoint;$('api-endpoint').textContent=endpoint?`请求地址：${endpoint}`:'地址检查通过。';updateMethodNotice();}
function apiFeedback(message,kind='success'){$('api-validation').textContent=message;$('api-validation').setAttribute('data-kind',kind);show('api-validation');announce(message);}
async function checkApi(event){event.preventDefault();if(state.connectionBusy||!validateApiFields())return;const version=state.configVersion,config=apiConfig();$('api-form-error').textContent='';$('api-check').disabled=true;show('api-validation',false);try{const data=await request('/api/api-check',{config});if(version!==state.configVersion){$('api-form-error').textContent='设置已改变，请重新应用。';return;}applyLocalCheck(data);apiFeedback('已应用 · 配置与运行目录检查通过，尚未验证服务连接。');}catch(err){uncheckApi();$('api-form-error').textContent=safeError(err);}finally{$('api-check').disabled=false;}}
async function checkModels(){
  if(state.connectionBusy||!validateApiFields())return;const version=state.configVersion,config=apiConfig();state.connectionBusy=true;$('api-test').disabled=true;$('api-check').disabled=true;$('api-test').textContent='正在检查…';$('api-form-error').textContent='';show('api-validation',false);
  try{const local=await request('/api/api-check',{config});if(version!==state.configVersion)return;applyLocalCheck(local);const data=await request('/api/api-models',{config});if(version!==state.configVersion)return;
    if(data.status==='models_unsupported'){state.modelCheck='unsupported';apiFeedback('服务未提供可用的模型列表，模型可用性未验证。可保留手填的模型 ID。','warning');}
    else if(data.selected_model_found===true){state.modelCheck='verified';apiFeedback('模型列表检查通过 · 已找到所选模型，未发送生成请求。');}
    else{state.modelCheck='missing';apiFeedback('服务模型列表中没有所选模型，请核对模型 ID；未发送生成请求。','warning');}updateMethodNotice();
  }catch(err){if(version===state.configVersion){state.modelCheck='unverified';$('api-form-error').textContent=safeError(err);updateMethodNotice();}}
  finally{state.connectionBusy=false;$('api-test').disabled=false;$('api-check').disabled=false;$('api-test').textContent='检查连接';if(version!==state.configVersion)$('api-form-error').textContent='设置已改变，本次检查结果已丢弃，请重新检查。';}
}
async function loadApiPresets(){try{const data=await request('/api/api-presets');const current=$('api-model').value;state.apiPresets=(data.models||[]).filter(item=>typeof item.id==='string'&&typeof item.label==='string');if(!state.apiPresets.length)throw Error('本地模型列表为空。');$('api-model').replaceChildren(...state.apiPresets.map(item=>new Option(item.label,item.id)));$('api-model').value=state.apiPresets.some(item=>item.id===current)?current:data.default_model;state.presetsReady=true;show('api-presets-note',false);globalThis.TeacherSelect?.refresh();}catch(err){$('api-presets-note').textContent=safeError(err);show('api-presets-note');}}
function chartSVG(title,width=580,height=280){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',title);svg.classList.add('chart');const t=document.createElementNS(ns,'title');t.textContent=title;svg.append(t);return svg;}
function svgNode(svg,tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=String(text);svg.append(n);return n;}
function plotCurves(curves,key,title){
  const svg=chartSVG(title,600,330),colors=['#285f50','#a48961','#6b8391'],left=52,right=22,top=20,bottom=76,w=600-left-right,h=330-top-bottom;
  const records=curves.flatMap(c=>c.records||[]).filter(r=>Number.isFinite(r[key]));if(!records.length)return el('p','暂无真实曲线记录。','help');
  const maxEpoch=Math.max(...records.map(r=>r.epoch)),max=key==='dev_f1'?1:Math.ceil(Math.max(...records.map(r=>r[key]))*10)/10;
  for(let i=0;i<=4;i++){const y=top+h-i*h/4;svgNode(svg,'line',{x1:left,y1:y,x2:600-right,y2:y,class:'grid'});svgNode(svg,'text',{x:left-9,y:y+4,'text-anchor':'end'},key==='dev_f1'?`${i*25}%`:(max*i/4).toFixed(2));}
  svgNode(svg,'line',{x1:left,y1:top+h,x2:600-right,y2:top+h,class:'axis'});
  [1,3,5,7,10].filter(e=>e<=maxEpoch).forEach(epoch=>svgNode(svg,'text',{x:left+(epoch-1)/Math.max(1,maxEpoch-1)*w,y:top+h+23,'text-anchor':'middle'},epoch));
  svgNode(svg,'text',{x:600-right,y:top+h+45,'text-anchor':'end'},'训练轮次');
  curves.forEach((series,i)=>{const valid=(series.records||[]).filter(r=>Number.isFinite(r[key])),points=valid.map(r=>`${left+(r.epoch-1)/Math.max(1,maxEpoch-1)*w},${top+h-r[key]/max*h}`).join(' '),color=colors[i%colors.length];
    svgNode(svg,'polyline',{points,fill:'none',stroke:color,'stroke-width':2.3,'stroke-dasharray':series.rope===false?'6 4':'none'});
    valid.forEach(r=>svgNode(svg,'circle',{cx:left+(r.epoch-1)/Math.max(1,maxEpoch-1)*w,cy:top+h-r[key]/max*h,r:3,fill:color}));
    const x=left+(i%2)*270,y=310+Math.floor(i/2)*17;svgNode(svg,'line',{x1:x,y1:y-4,x2:x+20,y2:y-4,stroke:color,'stroke-width':2.3,'stroke-dasharray':series.rope===false?'6 4':'none'});svgNode(svg,'text',{x:x+27,y},series.method);
  });return svg;
}
function plotBars(items,title){const svg=chartSVG(title,580,Math.max(200,items.length*58+55)),height=Math.max(200,items.length*58+55),left=125,w=580-left-55;items.forEach((item,i)=>{const y=25+i*58;svgNode(svg,'text',{x:left-12,y:y+18,'text-anchor':'end'},item.label);svgNode(svg,'rect',{x:left,y,width:w,height:25,fill:'#edf2e8',rx:2});svgNode(svg,'rect',{x:left,y,width:Math.max(0,item.value||0)*w,height:25,fill:item.color||'#578769',rx:2});svgNode(svg,'text',{x:Math.min(565,left+Math.max(0,item.value||0)*w+7),y:y+18,class:'value'},fmt(item.value));});svgNode(svg,'text',{x:left,y:height-12},'严格 F1，横轴从 0 开始');return svg;}
function renderExperiments(){
  const data=state.project?.experiments;if(!data)return;const host=$('experiments-content');host.replaceChildren();
  const nav=el('div',null,'experiment-tabs'),panel=el('div');nav.setAttribute('aria-label','实验结果分类');host.append(nav,panel);const buttons=[];
  function section(title){const area=el('section',null,'experiment-section');area.append(el('h3',title));return area;}
  function figure(name,title){const details=el('details');details.append(el('summary',title));const image=el('img',null,'study-chart');image.src=`/report-source/figures/${name}.svg`;image.alt=title;image.loading='lazy';const link=el('a','查看原尺寸图','figure-link');link.href=image.src;link.target='_blank';link.rel='noopener';details.append(image,link);return details;}
  function metricTable(rows){return table(['方法',numeric('有效 / 尝试'),numeric('精确率'),numeric('召回率'),numeric('严格 F1')],(rows||[]).map(r=>[methods[r.method]||r.method,`${r.valid} / ${r.total}`,fmt(r.precision),fmt(r.recall),fmt(r.f1)]),'formal-table');}
  function drawFormal(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','完整 5000 条'),badge('正式冻结 · 主种子 20261019'));panel.append(head,metricTable(data.holdout5000));
    const common=el('details');common.append(el('summary','共同 500 条 · 五路线对照'),metricTable(data.common500),el('p','共同500是完整5000的固定子集。L0/L3分别有无效输出与传输失败，完整主指标保持N/A；不能把有效子集成绩当完整成绩。','scope-note'));panel.append(common);
    const multi=section('三种子稳定性');multi.append(table(['方法','范围',numeric('平均 F1'),numeric('样本标准差 / 百分点')],(data.multi_seed||[]).map(r=>[methods[r.method]||r.method,r.split==='common500'?'共同500':'完整5000',fmt(r.mean),Number.isFinite(r.sd)?(r.sd*100).toFixed(3):'N/A'])));panel.append(multi);
    panel.append(el('p','严格实体 Micro P/R/F1：边界与类别同时匹配。三个种子全部报告，标准差采用 ddof=1；固定主种子不因测试成绩更换。','scope-note'));
  }
  function drawDevelopment(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','学习率与 RoPE'),badge('内部开发集 · 1503 条'));panel.append(head);
    panel.append(el('p','测试公开后的开发补充研究。仅使用既有训练/内部开发记录，不替换正式权重、默认阈值或冻结测试成绩。','scope-note'));
    const grid=data.learning_rate_rope_grid||[];
    if(grid.length)panel.append(table([numeric('学习率'),numeric('RoPE 开 · 最佳 F1'),numeric('RoPE 关 · 最佳 F1'),numeric('配对差 / 百分点')],grid.map(r=>[String(r.encoder_lr),fmt(r.on),fmt(r.off),(r.delta_on_minus_off*100).toFixed(3)])));
    const stats=data.paired_statistics;
    if(stats?.delta_best){panel.append(el('p',`5e-5 · 三配对种子：最佳轮 F1 差 ${ (stats.delta_best.mean*100).toFixed(3)} ± ${(stats.delta_best.sample_sd*100).toFixed(3)} 个百分点；固定第3轮差 ${(stats.delta_fixed3.mean*100).toFixed(3)} ± ${(stats.delta_fixed3.sample_sd*100).toFixed(3)} 个百分点（样本标准差）。`,'study-stats'));}
    const paired=el('details');paired.append(el('summary','三个种子的配对明细'),table([numeric('种子'),numeric('开 · 最佳 F1'),numeric('关 · 最佳 F1'),numeric('最佳轮差 / 百分点'),numeric('第3轮差 / 百分点')],(data.paired_seeds||[]).map(r=>[String(r.seed),fmt(r.on_best),fmt(r.off_best),(r.delta_best*100).toFixed(3),(r.delta_fixed3*100).toFixed(3)])));panel.append(paired);
    const training=data.training_curves||[];
    const curveSection=section(`训练曲线 · ${training.length} 个独立任务`),pickerWrap=el('div',null,'training-picker'),pickerLabel=el('label','选择一组 RoPE 开 / 关对照'),picker=el('select'),chartHost=el('div',null,'experiment-grid');pickerLabel.htmlFor='training-picker';picker.id='training-picker';picker.setAttribute('aria-label','选择训练曲线对照');
    const pairs=[...new Map(training.map(r=>[`${r.lr}|${r.seed}`,{lr:r.lr,seed:r.seed}])).values()];
    pairs.forEach((p,i)=>picker.append(new Option(`${p.lr.toExponential(0)} · 种子 ${p.seed}`,String(i))));picker.value='0';pickerWrap.append(pickerLabel,picker);curveSection.append(pickerWrap,chartHost);
    function drawCurves(){const choice=pairs[Number(picker.value)];chartHost.replaceChildren();if(!choice){chartHost.append(el('p','补充研究曲线尚未随此版本提供。','help'));return;}
      const curves=training.filter(r=>r.lr===choice.lr&&r.seed===choice.seed).sort((a,b)=>Number(b.rope)-Number(a.rope)).map(r=>({...r,method:`RoPE ${r.rope?'开':'关'} · ${r.seed}`}));
      [['训练损失','train_loss'],['开发集严格 F1','dev_f1']].forEach(([title,key])=>{const area=el('section',null,'chart-area');area.append(el('h4',title),plotCurves(curves,key,title));chartHost.append(area);});
    }
    picker.onchange=drawCurves;drawCurves();panel.append(curveSection);
    const all=el('details');all.append(el('summary','全部10组任务与数据表'),table([numeric('学习率'),numeric('种子'),centered('RoPE'),numeric('最佳轮'),numeric('最佳开发 F1'),numeric('第3轮 F1')],training.map(r=>[String(r.lr),String(r.seed),r.rope?'开':'关',String(r.best_epoch),fmt(r.best),fmt(r.third)])));
    training.forEach(r=>{const detail=el('details');detail.append(el('summary',`${r.lr} · ${r.seed} · RoPE ${r.rope?'开':'关'} · 逐轮数据`),table([numeric('轮次'),numeric('训练损失'),numeric('开发 F1')],(r.records||[]).map(row=>[String(row.epoch),row.train_loss.toFixed(4),fmt(row.dev_f1)])));all.append(detail);});panel.append(all,figure('rope-paired-effects','完整学习率与种子配对分析图'));
    const thresholds=section('解码阈值敏感性');thresholds.append(table(['方法',numeric('阈值'),numeric('精确率'),numeric('召回率'),numeric('严格 F1')],(data.threshold||[]).map(r=>[r.method,String(r.threshold),fmt(r.precision),fmt(r.recall),fmt(r.f1)])),el('p','正式默认阈值仍为0；此表只展示已保存的开发实验。','scope-note'),figure('threshold-tradeoff','阈值取舍图'));panel.append(thresholds);
  }
  function drawErrors(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','N1 错误分层'),badge('共同500 · 冻结预测'));panel.append(head);
    const grid=el('div',null,'experiment-grid'),groups=section('遗漏实体分组'),bars=el('div',null,'error-bars'),values=data.error_groups||[],max=Math.max(1,...values.map(g=>g.count));
    values.forEach(g=>{const row=el('div',null,'error-bar-row'),track=el('div',null,'error-bar-track'),fill=el('div',null,'error-bar-fill');fill.style.width=`${g.count/max*100}%`;track.append(fill);row.append(el('span',g.label),track,el('span',g.count,'mono'));bars.append(row);});groups.append(bars);
    if(data.error_counts)groups.append(el('p',`参考 ${data.error_counts.gold} · TP ${data.error_counts.tp} · FP ${data.error_counts.fp} · FN ${data.error_counts.fn}`,'scope-note'));
    const stratified=section('嵌套与未登录实体');stratified.append(table(['范围',numeric('召回率')],[['嵌套实体',fmt(data.nested_recall)],['未登录实体（OOV）',fmt(data.oov_recall)]]));grid.append(groups,stratified);panel.append(grid);
    if(data.by_type){const details=section('九类实体');details.append(table(['类别',numeric('参考数'),numeric('精确率'),numeric('召回率'),numeric('F1')],Object.entries(data.by_type).map(([key,v])=>[names[key]||key,v.gold,fmt(v.precision),fmt(v.recall),fmt(v.f1)])));panel.append(details);}
    panel.append(el('p','错误归组是输出观察，不是对模型内部机制的因果证明。','scope-note'));
  }
  function draw(tab){state.experimentTab=tab;buttons.forEach(([id,b])=>b.setAttribute('aria-pressed',String(id===tab)));panel.replaceChildren();({formal:drawFormal,development:drawDevelopment,errors:drawErrors}[tab]||drawFormal)();
    const source=el('details',null,'source-notes');source.append(el('summary','结果来源与范围'));if(state.project.generated_at)source.append(el('p',`展示摘要生成：${formatDate(state.project.generated_at)}`));(data.notes||[]).forEach(note=>source.append(el('p',note)));(data.sources||[]).forEach(path=>source.append(el('p',path,'mono')));panel.append(source);if(state.view==='experiments')reveal(panel);
  }
  [['formal','正式冻结评估'],['development','开发参数研究'],['errors','错误分层']].forEach(([id,label])=>{const b=el('button',label);b.type='button';b.onclick=()=>draw(id);buttons.push([id,b]);nav.append(b);});draw(state.experimentTab);
}
function renderAssets(){const assets=state.project?.assets||{},host=$('asset-list');host.replaceChildren();const titles={data:'官方数据',archive:'原始压缩包',D0:'D0 词典',N0:'N0 权重',N1:'N1 权重',L3:'三样本示例'};Object.entries(assets).forEach(([key,value])=>{const row=el('div',null,'asset-row');row.append(el('strong',titles[key]||key),el('span',value.ready?'已就绪':'缺失',`asset-status ${value.ready?'':'missing'}`),el('span',value.message||''));host.append(row);});const needed=['data','N0','N1'];const missing=needed.filter(key=>assets[key]?.ready===false);$('repro-availability').textContent=missing.length?`待准备：${missing.map(k=>titles[k]).join('、')}。缺资产时入口将明确报告。`:'将创建新目录，不覆盖已有预测；同一时间只运行一个完整复现。';}
function renderHistory(){
  const history=$('repro-history');history.replaceChildren();const jobs=state.savedJobs||[];
  jobs.forEach(job=>{const row=el('div',null,'history-record'),labels={passed:'通过',failed:'失败',cancelled:'已取消'};row.append(el('strong',`已保存任务 · ${labels[job.status]||job.status}`),el('p',`${formatDate(job.started_at)} · ${seconds(job.seconds)} · ${job.output||''}`));
    if(job.summary){const summary=summaryDTO(job.summary),detail=el('details');detail.append(el('summary','查看历史核对结果'),reproductionTable(summary));row.append(detail);}history.append(row);
  });
  (state.project?.previous_reproductions||[]).forEach(record=>{const row=el('div',null,'history-record'),label=String(record.label||'已保存的 CPU 重评').replace('候选包解压后CPU重评','独立环境 CPU 重评');row.append(el('strong',label),el('p',`${formatDate(record.at)} · ${seconds(record.seconds)} · 历史 ${record.status}`),el('p',record.source,'mono'));history.append(row);});
  if(!history.children.length)history.append(el('p','尚无已保存的历史记录。','help'));
}
function reproductionTable(summary){
  return table(['路线',numeric('已评估样本'),numeric('严格 F1'),centered('冻结指标核对')],Object.entries(summary.results||{}).map(([method,row])=>[method,row.samples??'—',fmt(row.micro?.f1),row.frozen_metric_equal===true?'一致':row.frozen_metric_equal===false?'不一致':'未取得结果']));
}
function renderProject(raw){
  const project=projectDTO(raw);state.project=project;$('identity').textContent=project.people.map(person=>`${person.name} ${person.student_id}`).join('　/　');
  const samples=$('live-samples');samples.replaceChildren();(project.samples||[]).forEach(sample=>{
    const b=el('button',sample.label,'sample-button');b.type='button';b.setAttribute('aria-pressed','false');
    b.onclick=()=>{samples.querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');$('live-text').value=sample.text;invalidateLive('已载入示例，等待本次抽取。');
      const purpose=sample.id==='nested'?'观察长短跨度及嵌套漏检；参考对照可在错误分析 F04 查看。':sample.id==='emoji'?'坐标反例：检查特殊字符后的定位，也观察模型可能出现的误识别。':'';
      $('sample-purpose').textContent=purpose;show('sample-purpose',!!purpose);$('live-text').focus();};samples.append(b);
  });renderAssets();renderExperiments();renderHistory();updateMethodNotice();
}
async function refreshProject(){try{const project=await request('/api/project');renderProject(project);show('global-error',false);}catch(err){$('global-error').textContent=`项目状态读取失败：${safeError(err)}`;show('global-error');}}
function renderJob(job){
  state.job=job;show('repro-job');show('repro-empty',false);const status=job.status||'running',running=status==='running'||status==='starting';
  $('repro-start').disabled=running;show('repro-cancel',running);$('repro-cancel').disabled=false;
  const labels={running:'运行中',starting:'启动中',passed:'本次复现通过',failed:'本次复现失败',cancelled:'本次已取消，部分输出保留'};
  $('repro-stage').textContent=`${state.restoredJobId===job.id?'恢复的任务 · ':''}${labels[status]||status} · ${job.stage||''}`;$('repro-time').textContent=seconds(job.seconds);
  $('repro-output').textContent=`输出：${job.output||'正在创建新目录'}　开始：${formatDate(job.started_at)}${job.finished_at?`　结束：${formatDate(job.finished_at)}`:''}`;
  const p=job.progress;let progress=typeof p==='number'?(p<=1?p*100:p):Number(p?.percent||0);
  if(p&&p.method){const order={D0:0,N0:1,N1:2},base=order[p.method]??0,ratio=p.total_windows?Math.min(1,Number(p.completed_windows||0)/Number(p.total_windows)):p.frozen_metric_equal!==undefined?1:0;progress=(base+ratio)/3*100;}
  if(status==='passed')progress=100;$('repro-progress').value=progress;
  const log=Array.isArray(job.log)?job.log.join('\n'):typeof job.log==='object'?JSON.stringify(job.log,null,2):job.log||'',logBox=$('repro-log'),nearBottom=logBox.scrollTop+logBox.clientHeight>=logBox.scrollHeight-35;
  logBox.textContent=log;if(nearBottom)logBox.scrollTop=logBox.scrollHeight;$('repro-summary').replaceChildren();
  if(job.summary){const summary=summaryDTO(job.summary);$('repro-summary').append(reproductionTable(summary));const details=el('details');details.append(el('summary','原始核对摘要 JSON'),el('pre',JSON.stringify(summary,null,2),'json-box'));$('repro-summary').append(details);}
  $('repro-error').textContent=job.error?(typeof job.error==='string'?job.error:JSON.stringify(job.error)):'';
  if(running)scheduleJobPoll(job.id);else if(state.poll){clearTimeout(state.poll);state.poll=null;}
  announce(`${labels[status]||status}。`,'delivery');
}
function scheduleJobPoll(id){if(state.poll)clearTimeout(state.poll);state.poll=setTimeout(async()=>{try{renderJob(await request(`/api/reproduction/jobs/${encodeURIComponent(id)}`));}catch(err){$('repro-error').textContent=`任务状态暂未读取：${safeError(err)}；正在重试。`;scheduleJobPoll(id);}},1500);}
async function restoreJobs(){
  try{const list=await request('/api/reproduction/jobs'),jobs=Array.isArray(list)?list:list.jobs||[];
    if(list.protection?.locked)$('repro-error').textContent=list.protection.message||'发现复现保护锁，需先核查原任务。';
    const active=list.active||jobs.find(job=>job.status==='running');state.savedJobs=jobs.filter(job=>job.id!==active?.id&&job.id!==state.job?.id);renderHistory();
    if(active){state.restoredJobId=active.id;renderJob(active);}
    else if(!state.job){show('repro-job',false);show('repro-empty');$('repro-start').disabled=false;}
  }catch(err){$('repro-error').textContent=safeError(err);}
}
async function startRepro(){$('repro-error').textContent='';$('repro-start').disabled=true;try{renderJob(await request('/api/reproduction/jobs',{}));announce('已启动本次完整 5000 条 CPU 复现。','delivery');}catch(err){$('repro-error').textContent=safeError(err);$('repro-start').disabled=false;}}
async function cancelRepro(){if(!state.job)return;$('repro-cancel').disabled=true;try{renderJob(await request(`/api/reproduction/jobs/${encodeURIComponent(state.job.id)}/cancel`,{}));}catch(err){$('repro-error').textContent=safeError(err);$('repro-cancel').disabled=false;}}
function navigate(view){
  globalThis.TeacherSelect?.close();
  if(view===state.view)return;state.view=view;
  document.querySelectorAll('.view').forEach(section=>section.hidden=section.id!==`view-${view}`);
  document.querySelectorAll('.main-nav button').forEach(button=>{if(button.dataset.view===view)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  if(view==='delivery')restoreJobs();
  if(view==='compare'&&!state.compare.result&&!state.compare.busy){$('compare-text').value=$('live-text').value;updateCounts();}
  if(state.compare.busy&&view!=='compare')invalidateCompare();
  $('footer-status').textContent=state.messages[view]||'';updateRuntime();$('main').focus({preventScroll:true});reveal($('view-'+view));
}
document.querySelectorAll('.main-nav button').forEach(button=>button.onclick=()=>navigate(button.dataset.view));$('open-api').onclick=openApi;$('close-api').onclick=closeApi;$('api-dialog').addEventListener('close',()=>{$('open-api').focus();});$('api-dialog').addEventListener('click',event=>{if(event.target===$('api-dialog')){const rect=$('api-dialog').getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)closeApi();}});$('api-form').onsubmit=checkApi;$('api-test').onclick=checkModels;document.querySelectorAll('[data-provider]').forEach(button=>button.onclick=()=>{if(button.dataset.provider!==$('api-provider').value){$('api-provider').value=button.dataset.provider;applyProvider();}});$('api-provider').onchange=applyProvider;$('api-base').addEventListener('input',changeAddress);$('api-model').addEventListener('change',()=>{$('api-json').checked=state.apiPresets.find(item=>item.id===$('api-model').value)?.json_mode!==false;uncheckApi();});$('api-form').addEventListener('invalid',event=>{focusInvalid(event.target,'请检查此字段。');},true);Array.from($('api-form').querySelectorAll('input,select')).forEach(field=>field.addEventListener('input',()=>{field.removeAttribute('aria-invalid');uncheckApi();}));$('api-clear-key').onclick=()=>{$('api-key').value='';uncheckApi();announce('API 密钥已从当前页面清除。');};$('live-run').onclick=runLive;$('live-text').oninput=()=>{invalidateLive();show('sample-purpose',false);$('live-samples').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed','false'));};$('live-method').onchange=()=>{invalidateLive();updateMethodNotice();};$('live-clear').onclick=()=>{show('sample-purpose',false);$('live-text').value='';invalidateLive('已清空原文，请输入待抽取文本。');$('live-text').focus();};$('live-text').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();runLive();}});$('live-json').onclick=()=>{if(!state.live.result)return;show('live-json-details');$('live-json-details').open=!$('live-json-details').open;};$('live-copy').onclick=copyJson;$('live-export').onclick=()=>exportJson(state.live.result,`医学实体_${state.live.result?.method||'结果'}_现场.json`);$('compare-text').oninput=invalidateCompare;document.querySelectorAll('.compare-method').forEach(input=>input.onchange=invalidateCompare);$('compare-run').onclick=runCompare;$('compare-export').onclick=()=>exportJson(state.compare.result,'医学实体_当前方法对比.json');$('case-select').onchange=loadCase;$('case-route').onchange=renderCase;$('case-export').onclick=()=>exportJson(state.cases.result,`医学实体_${state.cases.result?.selected_case}_${state.cases.result?.selected_route}_历史证据.json`);$('refresh-project').onclick=refreshProject;$('repro-start').onclick=startRepro;$('repro-cancel').onclick=cancelRepro;

$('case-only-diff').onchange=renderCase;
$('live-help-toggle').onclick=()=>{const expanded=$('live-guide').hidden;show('live-guide',expanded);$('live-help-toggle').setAttribute('aria-expanded',String(expanded));};
$('live-expand').onclick=()=>{const expanded=$('live-expand').getAttribute('aria-expanded')!=='true';$('live-expand').setAttribute('aria-expanded',String(expanded));$('live-expand').textContent=expanded?'收起输入':'展开输入';if(expanded)$('live-text').classList.add('expanded');else $('live-text').classList.remove('expanded');};
document.addEventListener?.('keydown',()=>{state.keyboard=true;document.body.setAttribute('data-keyboard','true');},true);
document.addEventListener?.('pointerdown',()=>{state.keyboard=false;document.body.setAttribute('data-keyboard','false');},true);

window.addEventListener('pagehide',()=>{$('api-key').value='';uncheckApi();});
updateApiLayout();updateCounts();refreshProject();loadCases();loadApiPresets();globalThis.TeacherSelect?.refresh();
})();
