'use strict';
(() => {
const $ = id => document.getElementById(id);
const names = {dis:'疾病',sym:'临床表现',dru:'药物',equ:'医疗设备',pro:'医疗过程',bod:'身体部位与物质',ite:'医学检验',mic:'微生物',dep:'科室'};
const methods = {D0:'D0 · 词典匹配',N0:'N0 · 字符模型',N1:'N1 · BERT 实体抽取',L0:'API · 零样本',L3:'API · 三样本'};
const state = {view:'live',project:null,apiPresets:[],apiProviders:[],apiModels:[],apiPreferences:{},presetsReady:false,apiModelManual:false,apiModelListingUnavailable:false,configChecked:false,configVersion:0,modelCheck:'unverified',connectionBusy:false,live:{version:0,busy:false,result:null},compare:{version:0,busy:false,result:null},cases:{version:0,data:null,result:null},job:null,poll:null,apiBusy:false,messages:{},experimentTab:'formal',experimentSelection:null,compareOnlyDiff:false,keyboard:false};
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
  const body=el('tbody');rows.forEach(row=>{const r=el('tr',null,row.className);row.forEach((v,i)=>{const td=el('td',null,`align-${columns[i]?.align||'left'}`);if(v instanceof Node)td.append(v);else td.textContent=v===null||v===undefined?'—':String(v);r.append(td);});body.append(r);});t.append(body);wrap.append(t);return wrap;
}
function clearLiveActions(){['live-json','live-copy','live-export'].forEach(id=>$(id).disabled=true);setLiveJsonVisible(false);$('live-json-content').textContent='';}
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
    if(isApi(method)){kind='api';label=state.live.busy?'API · 请求中':state.live.result?.status==='valid'?'API · 本次请求完成':state.configChecked?(state.modelCheck==='verified'?'API · 已获取模型目录':'API · 已本地检查'):'API · 未配置';}
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

function userError(message){const text=typeof message==='string'?message:JSON.stringify(message);return /预算|费用|预留|额度|余额|[¥￥]|\d+(?:\.\d+)?\s*元|budget|billing|balance|credit|cost|reservation/i.test(text)?'服务商账户当前不可用，请检查服务商返回的状态及账户余额。':text;}
function entityDTO(entity){return {start:entity.start,end:entity.end,type:entity.type,text:entity.text};}
function resultDTO(data){const dto={};for(const key of ['request_id','status','text','method','seconds','provider','model','endpoint','source','new_api_calls','started_at','finished_at','http_status','prompt_mode','raw_output','error_code','stage','request_sent'])if(data[key]!==undefined)dto[key]=data[key];dto.prediction=Array.isArray(data.prediction)?data.prediction.map(entityDTO):null;if(data.neural_timing){dto.neural_timing={};for(const key of ['device','precision','windows','load_seconds','inference_seconds'])if(data.neural_timing[key]!==undefined)dto.neural_timing[key]=data.neural_timing[key];}if(data.error)dto.error=userError(data.error);if(data.cost)dto.cost={state:data.cost.state,estimated_cny:data.cost.estimated_cny??null,is_invoice:false,price_source:data.cost.price_source??null};if(data.usage){dto.usage={};for(const key of ['prompt_tokens','completion_tokens','total_tokens','input_tokens','output_tokens'])if(data.usage[key]!==undefined)dto.usage[key]=data.usage[key];}return dto;}
function projectDTO(data){return {title:data.title,generated_at:data.generated_at,people:(data.people||[]).map(person=>({name:person.name,student_id:person.student_id})),samples:data.samples||[],assets:data.assets||{},experiments:data.experiments||{},artifacts:(data.artifacts||[]).map(item=>({id:item.id,label:item.label,available:item.available,url:item.url,status:item.status,file_name:item.file_name,file_size:item.file_size,modified_at:item.modified_at})),previous_reproductions:data.previous_reproductions||[],links:data.links||{}};}
function summaryDTO(data){const result=data.result||data,output={};for(const key of ['status','split','samples','started_at','finished_at','seconds'])if(data[key]!==undefined)output[key]=data[key];output.results={};for(const [method,row] of Object.entries(result.results||{})){const metrics=row.metrics||row;output.results[method]={samples:metrics.samples,micro:metrics.micro,frozen_metric_equal:row.frozen_metric_equal};}return output;}
function currentApiModelId(){return (state.apiModelManual?$('api-model-manual').value:$('api-model').value).trim();}
function selectedApiModel(){return state.apiModels.find(item=>item.id===currentApiModelId())||null;}
function hasModelPrice(item){return !!item&&!!item.price_source&&item.input_price_cny!=null&&item.output_price_cny!=null;}
function apiConfig(){const provider=$('api-provider').value;return {provider,model:currentApiModelId(),api_key:$('api-key').value.trim(),json_mode:true,...(provider==='custom'?{base_url:$('api-base').value.trim()}:{})};}
function requireApiConfig(){const config=apiConfig();if(!config.api_key.trim())throw Error('请先在右上角“API 设置”选择模型、填写密钥并保存。');if(!state.configChecked)throw Error('API 设置尚未保存，请在“API 设置”中点击保存并使用。');return config;}
function requestId(){return globalThis.crypto?.randomUUID?crypto.randomUUID():`teacher-${Date.now()}-${Math.random().toString(16).slice(2)}`;}
async function extract(text,method,apiSnapshot=null){if(!isApi(method)){const d=await request('/api/extract',{text,method});return resultDTO({...d,text,method,source:d.source||'teacher_live_cpu',status:d.status||'valid'});}const config=apiSnapshot||requireApiConfig();if(state.apiBusy)throw Error('已有现场 API 请求正在运行，请等待它完成。');if(method==='L3'&&state.project?.assets?.L3?.ready===false)throw Error('三样本所需官方训练数据尚未准备，请先准备资产。');state.apiBusy=true;try{const d=await request('/api/extract-api',{request_id:requestId(),text,prompt_mode:method,config});return resultDTO({...d,text,method});}finally{state.apiBusy=false;}}
function liveMetadata(data){
  const wrap=el('div'),meta=el('div',null,'result-meta');meta.append(el('span',methods[data.method]||data.method),el('span',`· ${sourceLabel(data.source)}`),el('span',`· 本次总耗时 ${seconds(data.seconds)}`));
  if(isApi(data.method)){if(data.model)meta.append(el('span',`· ${data.model}`));const usage=data.usage;meta.append(el('span',`输入 ${usage?.prompt_tokens??usage?.input_tokens??'未知'} / 输出 ${usage?.completion_tokens??usage?.output_tokens??'未知'} token`));meta.append(el('span',data.cost?.estimated_cny!=null?`参考费用 ¥${data.cost.estimated_cny}（非账单）`:'参考费用：未知')); }
  wrap.append(meta);if(data.neural_timing){const detail=el('details',null,'timing-note'),t=data.neural_timing;detail.append(el('summary','推理计时详情'),el('p',`本次模型推理 ${seconds(t.inference_seconds)} · ${t.windows} 个窗口。模型在本服务首次加载 ${seconds(t.load_seconds)}，此项不是每次重复加载；现场总耗时不等于实验平均延迟。`));wrap.append(detail);}return wrap;
}
function setLiveBusy(busy,method){state.live.busy=busy;$('live-run').disabled=busy;$('live-result').setAttribute('aria-busy',String(busy));show('live-progress',busy);updateRuntime();if(busy){$('live-progress').textContent=`正在运行 ${methods[method]}${isApi(method)?' · 等待服务响应':' · 本机 CPU 加载模型并推理'}，0.0 秒`;const began=performance.now();const timer=setInterval(()=>{if(!state.live.busy)return clearInterval(timer);$('live-progress').textContent=`正在运行 ${methods[method]}${isApi(method)?' · 等待服务响应':' · 本机 CPU 推理'}，${((performance.now()-began)/1000).toFixed(1)} 秒`;},250);}}
async function runLive(){if(state.live.busy)return;const text=$('live-text').value,method=$('live-method').value;try{validateText(text,'live-text');if(isApi(method))requireApiConfig();}catch(err){$('live-error').textContent=safeError(err);if(isApi(method)&&!state.configChecked)openApi();return;}invalidateLive('正在运行，请等待本次抽取结果…');const version=state.live.version;setLiveBusy(true,method);$('live-result-state').textContent='运行中';$('live-result-state').className='badge';try{const data=await extract(text,method);if(version!==state.live.version)return;state.live.result={...data,export_schema:'teacher-entity-result-v1',selected_route:method,text,exported_from:'实时抽取',obtained_at:new Date().toISOString()};$('live-result').replaceChildren(liveMetadata(data),entityView(text,data.prediction,{status:data.status,raw:data.raw_output??data.raw_response,error:data.error,label:methods[method]}));$('live-result-state').textContent=data.status==='valid'?'已完成':'保留失败';$('live-result-state').className=data.status==='valid'?'badge':'badge failure';$('live-json-content').textContent=JSON.stringify(state.live.result,null,2);['live-json','live-copy','live-export'].forEach(id=>$(id).disabled=false);announce(data.status==='valid'?'本次抽取完成，可定位或导出实体。':'本次输出未通过校验，失败原文已保留。','live');updateRuntime();}catch(err){if(version===state.live.version){$('live-error').textContent=safeError(err);$('live-result').replaceChildren(empty('运行未完成，请检查错误后重试。'));$('live-result-state').textContent='运行失败';$('live-result-state').className='badge failure';}}finally{setLiveBusy(false,method);}}
function updateMethodNotice(){
  const method=$('live-method').value;show('live-api-note',isApi(method));
  $('live-api-note').textContent=state.configChecked?(state.modelCheck==='verified'?'已获取模型目录。开始抽取会发送当前文本，能否调用以本次响应为准。':'本地检查通过。开始抽取会将当前文本发送至所选服务。'):'尚未配置 API。请先打开右上角 API 设置；本地方法可直接运行。';
  const asset=state.project?.assets?.[method];$('live-asset-note').textContent=asset?.ready===false?asset.message||'所需资产尚未准备。':'';show('live-asset-note',asset?.ready===false);updateRuntime();
}
function interruptApiComparison(){
  const run=state.compare.run;
  if(!state.compare.busy||!run?.apiConfig||run.configVersion===state.configVersion)return;
  run.interrupted=true;
  $('compare-state').textContent='配置已改变，对比已中断';$('compare-state').className='badge warning';
  $('compare-progress').textContent='配置已改变；等待已发送请求返回，其余路线不再发送。';
  announce('配置已改变，对比已中断；已发送请求的真实结果将保留。','compare');
}
async function runCompare(){
  if(state.compare.busy)return;
  const text=$('compare-text').value,selected=Array.from(document.querySelectorAll('.compare-method:checked')).map(e=>e.value);
  let apiSnapshot=null;
  try{validateText(text,'compare-text');if(!selected.length)throw Error('请至少选择一种方法。');if(selected.some(isApi))apiSnapshot={...requireApiConfig()};}
  catch(err){$('compare-error').textContent=safeError(err);if(selected.some(isApi)&&!state.configChecked)openApi();return;}
  invalidateCompare();$('compare-result').replaceChildren(empty('正在运行所选方法，请等待本次对比结果…'));
  const version=state.compare.version,run={apiConfig:apiSnapshot,configVersion:state.configVersion,interrupted:false};
  state.compare.run=run;state.compare.busy=true;announce('正在运行所选方法，请等待本次对比结果。','compare');
  $('compare-run').disabled=true;$('compare-result').setAttribute('aria-busy','true');show('compare-progress');$('compare-state').textContent='运行中';$('compare-state').className='badge';
  const results=[];
  try{
    for(let i=0;i<selected.length;i++){
      if(version!==state.compare.version||run.interrupted)break;
      const method=selected[i];$('compare-progress').textContent=`${i+1} / ${selected.length} · 正在运行 ${methods[method]}${isApi(method)?'（将外发当前文本）':'（本机 CPU）'}`;
      try{results.push(await extract(text,method,apiSnapshot));}
      catch(err){results.push({text,method,prediction:null,status:'request_failed',error:safeError(err),source:isApi(method)?'teacher_live_api':'teacher_live_cpu'});}
      if(version!==state.compare.version)break;
      renderComparison(text,results,selected,run);
    }
    if(version!==state.compare.version)return;
    const notRun=selected.filter(method=>!results.some(r=>r.method===method));
    state.compare.result={export_schema:'teacher-method-comparison-v1',text,selected_routes:selected,source:'本次现场同文本方法对比',no_reference:true,status:run.interrupted?'interrupted':'completed',not_run_routes:notRun,interruption_reason:run.interrupted?'api_config_changed':null,...(apiSnapshot?{api_configuration:{provider:apiSnapshot.provider,model:apiSnapshot.model}}:{}),results,obtained_at:new Date().toISOString()};
    renderComparison(text,results,selected,run);$('compare-export').disabled=false;
    const failed=results.filter(r=>r.status!=='valid').length;
    $('compare-state').textContent=run.interrupted?'配置已改变，对比已中断':failed?`已完成 · ${failed} 项失败`:'已完成';
    $('compare-state').className=run.interrupted||failed?'badge warning':'badge';
    announce(run.interrupted?`对比已中断：已返回 ${results.length} 项，未运行 ${notRun.length} 项。请按新配置重新运行。`:'方法对比完成；✓ 仅代表检出，不代表正确。','compare');
  }finally{state.compare.busy=false;state.compare.run=null;$('compare-run').disabled=false;$('compare-result').setAttribute('aria-busy','false');show('compare-progress',false);}
}
function renderComparison(text,results,selected,run=null){
  const host=$('compare-result');host.replaceChildren();const union=new Map();
  results.forEach(r=>(r.prediction||[]).forEach(entity=>{if(!union.has(entityKey(entity)))union.set(entityKey(entity),entity);}));
  const entities=[...union.values()].sort((a,b)=>a.start-b.start||a.end-b.end||a.type.localeCompare(b.type));
  const complete=selected.every(method=>results.some(r=>r.method===method&&r.status==='valid'));
  const present=(entity,method)=>results.find(r=>r.method===method)?.prediction?.some(e=>entityKey(e)===entityKey(entity));
  const common=complete?entities.filter(e=>selected.every(method=>present(e,method))):[];
  const summary=el('div',null,'comparison-summary');
  if(run?.apiConfig)summary.append(el('span',`${providerItem(run.apiConfig.provider)?.label||run.apiConfig.provider} · ${run.apiConfig.model}`,'comparison-model'));
  if(run?.interrupted)summary.append(badge('配置已改变，对比已中断','warning'));
  if(complete)summary.append(el('strong',`共同 ${common.length} 项`),el('span',`分歧 ${entities.length-common.length} 项${common.length===entities.length?' · 所选方法结果一致':''}`));
  selected.filter(method=>!results.some(r=>r.method===method)).forEach(method=>summary.append(el('span',`${method} · ${run?.interrupted?'未运行':'待运行'}`)));
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
      const row=[identity,`[${entity.start}, ${entity.end})`,...selected.map(method=>{const r=results.find(r=>r.method===method),v=el('span',!r?(run?.interrupted?'未运行':'待运行'):r.status!=='valid'?'失败':present(entity,method)?'✓':'—',r?.status==='valid'&&present(entity,method)?'check':'absent');v.title=!r?(run?.interrupted?'此路线未发送请求':'此路线尚未完成'):r.status!=='valid'?'此路线失败，不能按未检出处理':present(entity,method)?'该方法检出此精确跨度与类型':'该方法未检出此精确跨度与类型';return v;})];
      if(complete&&!commonKeys.has(entityKey(entity)))row.className='matrix-difference';return row;
    });
    if(rows.length)body.append(table(['实体 / 类别','Unicode 位置',...selected.map(method=>methods[method])],rows,'comparison-table'));
    else body.append(el('p',entities.length?'所选方法在当前文本上没有分歧。':complete?'所选方法均未检出实体；这不等同于模型正确。':'暂无有效实体，请查看各路线状态及原始输出。','comparison-message'));
    if(run?.interrupted)body.append(el('p','未运行路线未计入分歧或实体数量。请按新配置重新运行完整对比。','help'));
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
  const diff=$('case-diff');diff.replaceChildren();
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
function setLiveJsonVisible(visible){show('live-json-details',visible);$('live-json').textContent=visible?'收起 JSON':'查看 JSON';$('live-json').setAttribute('aria-expanded',String(visible));}
async function copyJson(){if(!state.live.result)return;const text=JSON.stringify(state.live.result,null,2);try{await navigator.clipboard.writeText(text);announce('已复制当前结果 JSON。');}catch{setLiveJsonVisible(true);$('live-error').textContent='浏览器未允许复制；可手动复制或导出 JSON。';}}
function hideApiKey(){$('api-key').type='password';$('api-key-toggle').setAttribute('aria-pressed','false');$('api-key-toggle').setAttribute('aria-label','显示 API 密钥');}
function openApi(){$('api-dialog').showModal();globalThis.TeacherSelect?.focus($('api-provider'));}
function closeApi(){globalThis.TeacherSelect?.close();$('api-dialog').close();$('open-api').focus();}
function providerItem(id=$('api-provider').value){return state.apiProviders.find(item=>item.id===id)||null;}
function readApiPreferences(){try{const value=JSON.parse(localStorage.getItem('teacher-api-preferences-v1')||'{}');return value&&typeof value==='object'&&!Array.isArray(value)?Object.fromEntries(['provider','model','base_url'].map(key=>[key,typeof value[key]==='string'?value[key]:''])):{};}catch{return {};}}
function persistApiPreferences(){if(typeof localStorage==='undefined')return;const prefs={provider:$('api-provider').value,model:currentApiModelId(),base_url:$('api-provider').value==='custom'?$('api-base').value.trim():''};try{localStorage.setItem('teacher-api-preferences-v1',JSON.stringify(prefs));}catch{}}
function destinationNote(){const item=providerItem();if(!item)return '本地服务目录尚未载入。';if(item.id==='custom')return '使用填写的 HTTPS 兼容接口地址。';try{return `${item.label} · ${new URL(item.base_url).hostname}`;}catch{return item.label;}}
function renderProviderOptions(preferred){const select=$('api-provider');select.replaceChildren(...state.apiProviders.map(item=>new Option(item.id==='deepseek'?`${item.label} · 默认`:item.label,item.id)));select.value=state.apiProviders.some(item=>item.id===preferred)?preferred:'deepseek';globalThis.TeacherSelect?.refresh();}
function renderApiModels(preferred){const select=$('api-model'),rows=state.apiModels,placeholder=new Option(rows.length?'请选择模型':'暂无模型，可刷新或手动输入','');const options=rows.map(item=>new Option(item.label||item.id,item.id));select.replaceChildren(placeholder,...options);select.value=(rows.find(item=>item.id===preferred)||rows.find(item=>item.id===providerItem()?.default_model)||rows[0])?.id||'';select.required=!state.apiModelManual;globalThis.TeacherSelect?.refresh();updateApiPricing();}
function setModelListUnavailable(unavailable){state.apiModelListingUnavailable=unavailable;$('api-model-manual-toggle').hidden=$('api-provider').value!=='custom'&&!unavailable;}
function setManualModelMode(manual){const previous=currentApiModelId();state.apiModelManual=manual;if(manual){if(!$('api-model-manual').value)$('api-model-manual').value=previous;}else{const options=$('api-model').options||$('api-model').children,option=[...options].find(item=>item.value===previous&&!item.disabled);if(option)$('api-model').value=previous;}$('api-model').required=!manual;$('api-model-manual').required=manual;show('api-model-select-wrap',!manual);show('api-model-manual-wrap',manual);$('api-model-manual-toggle').setAttribute('aria-pressed',String(manual));$('api-model-manual-toggle').textContent=manual?'从下拉列表选择':'手动输入模型 ID';updateApiPricing();}
function updateApiPricing(item=selectedApiModel()){if(!currentApiModelId()){show('api-price-summary',false);return;}$('api-price-summary').textContent=hasModelPrice(item)?`价格参考：输入 ¥${item.input_price_cny} / 输出 ¥${item.output_price_cny} 每百万 token，实际费用以服务商账单为准。`:'费用参考：未知；不影响使用，实际费用以服务商账单为准。';show('api-price-summary');}
function uncheckApi(){state.configVersion++;interruptApiComparison();state.configChecked=false;state.modelCheck='unverified';$('api-config-dot').classList.remove('ready');$('api-validation').hidden=true;$('api-form-error').textContent='';$('api-endpoint').textContent=destinationNote();updateMethodNotice();}
function updateApiLayout(){const custom=$('api-provider').value==='custom';show('api-base-wrap',custom);$('api-base').disabled=!custom;$('api-base').required=custom;setModelListUnavailable(state.apiModelListingUnavailable);updateApiPricing();$('api-endpoint').textContent=destinationNote();}
function applyProvider(){const provider=$('api-provider').value,item=providerItem(provider),same=state.apiPreferences.provider===provider,preferred=same?state.apiPreferences.model||'':item?.default_model||'';if(state.lastApiProvider&&state.lastApiProvider!==provider){$('api-key').value='';hideApiKey();}state.lastApiProvider=provider;state.apiModelManual=false;state.apiModelListingUnavailable=false;$('api-model').required=true;$('api-model-manual').required=false;$('api-model-manual').value='';show('api-model-manual-wrap',false);show('api-model-select-wrap');$('api-model-manual-toggle').setAttribute('aria-pressed','false');$('api-model-manual-toggle').textContent='手动输入模型 ID';$('api-base').value=provider==='custom'?(same&&typeof state.apiPreferences.base_url==='string'?state.apiPreferences.base_url:''):item?.base_url||'';state.apiModels=item?.models||[];renderApiModels(preferred);if(provider==='custom'&&(!state.apiModels.length||same&&preferred&&!state.apiModels.some(row=>row.id===preferred))){$('api-model-manual').value=same?preferred:'';setManualModelMode(true);}state.lastApiModel=currentApiModelId();updateApiLayout();persistApiPreferences();uncheckApi();}
function changeAddress(){persistApiPreferences();$('api-key').value='';hideApiKey();$('api-endpoint').textContent=destinationNote();uncheckApi();}
function changeApiModel(){state.lastApiModel=currentApiModelId();updateApiPricing();persistApiPreferences();uncheckApi();}
function changeManualModel(){changeApiModel();}
function focusInvalid(field,message){field.setAttribute('aria-invalid','true');$('api-form-error').textContent=message;if(field.tagName==='SELECT')globalThis.TeacherSelect?.focus(field);else field.focus();}
function validateModelListingFields(){if(!state.presetsReady){$('api-form-error').textContent='本地服务目录尚未就绪，请稍后重试。';return false;}if(!$('api-key').value.trim()){focusInvalid($('api-key'),'请先填写 API 密钥，再查询服务模型目录。');return false;}if($('api-provider').value==='custom'){const base=$('api-base');if(!base.value.trim()||base.validity?.valid===false){focusInvalid(base,'请输入公网 HTTPS 兼容接口地址。');return false;}}return true;}
function validateApiFields(){if(!state.presetsReady){$('api-form-error').textContent='本地服务目录尚未就绪，请稍后重试。';return false;}if(!currentApiModelId()){focusInvalid(state.apiModelManual?$('api-model-manual'):$('api-model'),'请选择或输入模型 ID。');return false;}if(!$('api-key').value.trim()){focusInvalid($('api-key'),'请填写 API 密钥。');return false;}if($('api-provider').value==='custom'){const base=$('api-base');if(!base.value.trim()||base.validity?.valid===false){focusInvalid(base,'请输入公网 HTTPS 兼容接口地址。');return false;}}return true;}
function applyLocalCheck(data){state.configChecked=true;$('api-config-dot').classList.add('ready');$('api-endpoint').textContent=destinationNote();persistApiPreferences();updateMethodNotice();}
function apiFeedback(message,kind='success'){$('api-validation').textContent=message;$('api-validation').setAttribute('data-kind',kind);show('api-validation');announce(message);}
async function checkApi(event){event.preventDefault();if(state.connectionBusy||$('api-check').disabled||!validateApiFields())return;const version=state.configVersion,config=apiConfig();$('api-form-error').textContent='';$('api-check').disabled=true;show('api-validation',false);try{const data=await request('/api/api-check',{config});if(version!==state.configVersion){$('api-form-error').textContent='设置已改变，请重新保存。';return;}applyLocalCheck(data);closeApi();announce('API 设置已保存；运行时才会发送原文，服务连接尚未验证。');}catch(err){uncheckApi();$('api-form-error').textContent=safeError(err);}finally{$('api-check').disabled=false;}}
async function checkModels(){
  if(state.connectionBusy||!validateModelListingFields())return;const version=state.configVersion,provider=$('api-provider').value,config={provider,api_key:$('api-key').value.trim(),...(provider==='custom'?{base_url:$('api-base').value.trim()}:{})};let applied=false;state.connectionBusy=true;$('api-test').disabled=true;$('api-check').disabled=true;$('api-test').textContent='正在刷新…';$('api-form-error').textContent='';show('api-validation',false);
  try{const data=await request('/api/api-models',{config});if(version!==state.configVersion)return;const before=currentApiModelId();if(data.status==='models_unsupported'){setModelListUnavailable(true);state.modelCheck='unsupported';apiFeedback('服务未提供模型目录，已保留现有选项。未发送待抽取文本。','warning');}else{const local=providerItem()?.models||[],remote=Array.isArray(data.models)?data.models:[],merged=new Map(state.apiModels.map(item=>[item.id,item]));for(const item of [...local,...remote])merged.set(item.id,{...merged.get(item.id),...item});state.apiModels=[...merged.values()];setModelListUnavailable(!remote.length);state.modelCheck='verified';if(state.apiModelManual&&remote.some(item=>item.id===before))setManualModelMode(false);renderApiModels(before);apiFeedback(`已获取 ${remote.length} 个模型，内置选项保留。未发送待抽取文本；能否调用以运行结果为准。`,remote.length?'success':'warning');}if(before!==currentApiModelId()){uncheckApi();persistApiPreferences();}applied=true;updateMethodNotice();
  }catch(err){if(version===state.configVersion){setModelListUnavailable(true);state.modelCheck='unverified';apiFeedback('刷新失败，现有模型仍可保存使用。未发送待抽取文本。','warning');$('api-form-error').textContent=safeError(err);updateMethodNotice();}}
  finally{state.connectionBusy=false;$('api-test').disabled=false;$('api-check').disabled=false;$('api-test').textContent='刷新模型';if(version!==state.configVersion&&!applied)$('api-form-error').textContent='设置已改变，本次目录结果已丢弃，请重新刷新。';}
}
async function loadApiPresets(){try{const data=await request('/api/api-presets');state.apiPreferences=readApiPreferences();state.apiProviders=Array.isArray(data.providers)?data.providers.filter(item=>item&&typeof item.id==='string'):[];if(!state.apiProviders.length){state.apiProviders=[{id:'deepseek',label:'DeepSeek',protocol:'openai_compatible',base_url:data.base_url,default_model:data.default_model,models:data.models||[]},{id:'custom',label:'OpenAI 兼容自定义',protocol:'openai_compatible',base_url:'',models:[]}];}state.apiPresets=(data.models||[]).filter(item=>typeof item.id==='string'&&typeof item.label==='string');const preferred=state.apiProviders.some(item=>item.id===state.apiPreferences.provider)?state.apiPreferences.provider:'deepseek';state.presetsReady=true;state.lastApiProvider='';renderProviderOptions(preferred);applyProvider();state.lastApiModel=currentApiModelId();updateApiLayout();}catch(err){$('api-form-error').textContent=`本地服务目录加载失败：${safeError(err)}`;}}
function chartSVG(title,width=580,height=280){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',title);svg.classList.add('chart');const t=document.createElementNS(ns,'title');t.textContent=title;svg.append(t);return svg;}
function svgNode(svg,tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=String(text);svg.append(n);return n;}
function plotCurves(curves,key,title){
  const compact=globalThis.matchMedia?.('(max-width: 750px)')?.matches,width=compact?Math.max(210,Math.min(360,(globalThis.innerWidth||390)-106)):500,height=260;
  const svg=chartSVG(title,width,height),colors=['#285f50','#a48961','#6b8391'],left=44,right=14,top=16,bottom=76,w=width-left-right,h=height-top-bottom;
  const records=curves.flatMap(c=>c.records||[]).filter(r=>Number.isFinite(r[key]));if(!records.length)return el('p','暂无真实曲线记录。','help');
  const maxEpoch=Math.max(...records.map(r=>r.epoch)),max=key==='dev_f1'?1:Math.ceil(Math.max(...records.map(r=>r[key]))*10)/10;
  for(let i=0;i<=4;i++){const y=top+h-i*h/4;svgNode(svg,'line',{x1:left,y1:y,x2:width-right,y2:y,class:'grid'});svgNode(svg,'text',{x:left-9,y:y+4,'text-anchor':'end'},key==='dev_f1'?`${i*25}%`:(max*i/4).toFixed(2));}
  svgNode(svg,'line',{x1:left,y1:top+h,x2:width-right,y2:top+h,class:'axis'});
  (compact?[1,5,10]:[1,3,5,7,10]).filter(e=>e<=maxEpoch).forEach(epoch=>svgNode(svg,'text',{x:left+(epoch-1)/Math.max(1,maxEpoch-1)*w,y:top+h+23,'text-anchor':'middle'},epoch));
  svgNode(svg,'text',{x:width-right,y:top+h+45,'text-anchor':'end'},'训练轮次');
  curves.forEach((series,i)=>{const valid=(series.records||[]).filter(r=>Number.isFinite(r[key])),points=valid.map(r=>`${left+(r.epoch-1)/Math.max(1,maxEpoch-1)*w},${top+h-r[key]/max*h}`).join(' '),color=colors[i%colors.length];
    svgNode(svg,'polyline',{points,fill:'none',stroke:color,'stroke-width':2.3,'stroke-dasharray':series.rope===false?'6 4':'none'});
    valid.forEach(r=>svgNode(svg,'circle',{cx:left+(r.epoch-1)/Math.max(1,maxEpoch-1)*w,cy:top+h-r[key]/max*h,r:3,fill:color}));
    const x=left+(i%2)*(compact?92:210),y=height-13+Math.floor(i/2)*17;svgNode(svg,'line',{x1:x,y1:y-4,x2:x+20,y2:y-4,stroke:color,'stroke-width':2.3,'stroke-dasharray':series.rope===false?'6 4':'none'});svgNode(svg,'text',{x:x+27,y},series.method);
  });return svg;
}
function plotBars(items,title){const svg=chartSVG(title,580,Math.max(200,items.length*58+55)),height=Math.max(200,items.length*58+55),left=125,w=580-left-55;items.forEach((item,i)=>{const y=25+i*58;svgNode(svg,'text',{x:left-12,y:y+18,'text-anchor':'end'},item.label);svgNode(svg,'rect',{x:left,y,width:w,height:25,fill:'#edf2e8',rx:2});svgNode(svg,'rect',{x:left,y,width:Math.max(0,item.value||0)*w,height:25,fill:item.color||'#578769',rx:2});svgNode(svg,'text',{x:Math.min(565,left+Math.max(0,item.value||0)*w+7),y:y+18,class:'value'},fmt(item.value));});svgNode(svg,'text',{x:left,y:height-12},'严格 F1，横轴从 0 开始');return svg;}
function plotMetricBars(rows,title){
  const valid=(rows||[]).filter(row=>['precision','recall','f1'].some(key=>Number.isFinite(row[key])));
  if(!valid.length)return el('p','暂无可视化指标记录。','help');
  const compact=globalThis.matchMedia?.('(max-width: 750px)')?.matches,width=compact?Math.max(230,Math.min(420,(globalThis.innerWidth||390)-82)):580,left=compact?90:142,right=54,top=34,rowHeight=56,barHeight=9,barGap=14,height=top+valid.length*rowHeight+10,plotWidth=width-left-right;
  const svg=chartSVG(title,width,height);svg.classList.add('metric-comparison-chart');
  const ticks=compact?2:4;
  for(let tick=0;tick<=ticks;tick++){
    const x=left+plotWidth*tick/ticks;
    svgNode(svg,'line',{x1:x,y1:top-8,x2:x,y2:height-10,class:'grid'});
    svgNode(svg,'text',{x,y:top-14,'text-anchor':tick===0?'start':tick===ticks?'end':'middle'},`${tick*100/ticks}%`);
  }
  const metrics=[['precision','精确率','precision'],['recall','召回率','recall'],['f1','严格 F1','f1']];
  valid.forEach((row,index)=>{
    const rowTop=top+index*rowHeight;
    if(index%2===0)svgNode(svg,'rect',{x:0,y:rowTop-5,width,height:rowHeight-1,class:'metric-row-shade'});
    svgNode(svg,'text',{x:left-12,y:rowTop+barGap+13,'text-anchor':'end',class:'metric-method'},compact?(row.label==='身体部位与物质'?'部位与物质':row.label||({L0:'API 零样本',L3:'API 三样本'}[row.method]||row.method)):row.label||methods[row.method]||row.method);
    metrics.forEach(([key,label,kind],metricIndex)=>{
      const value=Number.isFinite(row[key])?Math.max(0,Math.min(1,row[key])):0,y=rowTop+metricIndex*barGap;
      const group=svgNode(svg,'g',{class:`metric-series metric-${kind}`});
      svgNode(group,'title',{},`${row.label||methods[row.method]||row.method} · ${label} ${fmt(row[key])}`);
      svgNode(group,'rect',{x:left,y,width:plotWidth,height:barHeight,rx:2,class:'metric-bar-track'});
      svgNode(group,'rect',{x:left,y,width:plotWidth*value,height:barHeight,rx:2,class:'metric-bar-fill'});
      svgNode(group,'text',{x:width-5,y:y+8,'text-anchor':'end',class:'metric-value'},fmt(row[key]));
    });
  });
  return svg;
}
function renderExperiments(){
  const data=state.project?.experiments;if(!data)return;const host=$('experiments-content');host.replaceChildren();
  const nav=el('div',null,'experiment-tabs'),panel=el('div');nav.setAttribute('aria-label','实验结果分类');host.append(nav,panel);const buttons=[];
  function section(title){const area=el('section',null,'experiment-section');area.append(el('h3',title));return area;}
  function figure(name,title){const area=el('section',null,'study-figure');area.append(el('h4',title));const image=el('img',null,'study-chart');image.src=`/report-source/figures/${name}.svg`;image.alt=title;image.loading='lazy';const link=el('a','查看原尺寸图','figure-link');link.href=image.src;link.target='_blank';link.rel='noopener';area.append(image,link);return area;}
  function tablePanel(title,content){const area=el('section',null,'experiment-table-panel');area.append(el('h4',title),content);return area;}
  function metricTable(rows){return table(['方法',numeric('有效 / 尝试'),numeric('精确率'),numeric('召回率'),numeric('严格 F1')],(rows||[]).map(r=>[methods[r.method]||r.method,`${r.valid} / ${r.total}`,fmt(r.precision),fmt(r.recall),fmt(r.f1)]),'formal-table');}
  function drawFormal(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','完整 5000 条'),badge('正式冻结 · 主种子 20261019'));
    const overview=el('div',null,'formal-overview'),chartArea=el('section',null,'chart-area metric-chart-area'),dataArea=el('section',null,'formal-data-area');
    const chartHead=el('div',null,'metric-chart-head');chartHead.append(el('h4','精确率 · 召回率 · 严格 F1'));
    const legend=el('div',null,'metric-legend');[['precision','精确率'],['recall','召回率'],['f1','严格 F1']].forEach(([key,label])=>{const item=el('span',null,'metric-legend-item');item.append(el('i',null,`metric-swatch metric-${key}`),el('span',label));legend.append(item);});
    chartArea.append(chartHead,legend,plotMetricBars(data.holdout5000,'完整5000条正式冻结评估：精确率、召回率与严格F1'));
    dataArea.append(el('h4','评估数据'),metricTable(data.holdout5000));overview.append(chartArea,dataArea);panel.append(head,overview);
    const supplementary=el('div',null,'experiment-grid experiment-table-grid');
    supplementary.append(tablePanel('共同 500 条 · 五路线对照',metricTable(data.common500)),tablePanel('三种子稳定性',table(['方法','范围',numeric('平均 F1'),numeric('样本标准差 / 百分点')],(data.multi_seed||[]).map(r=>[methods[r.method]||r.method,r.split==='common500'?'共同500':'完整5000',fmt(r.mean),Number.isFinite(r.sd)?(r.sd*100).toFixed(3):'N/A']))));
    supplementary.firstElementChild.append(el('p','共同500是完整5000的固定子集。L0/L3分别有无效输出与传输失败，完整主指标保持N/A；不能把有效子集成绩当完整成绩。','scope-note'));
    panel.append(supplementary);
    panel.append(el('p','严格实体 Micro P/R/F1：边界与类别同时匹配。三个种子全部报告，标准差采用 ddof=1；固定主种子不因测试成绩更换。','scope-note'));
  }
  function drawDevelopment(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','学习率与 RoPE'),badge('内部开发集 · 1503 条'));panel.append(head);
    panel.append(el('p','测试公开后的开发补充研究。仅使用既有训练/内部开发记录，不替换正式权重、默认阈值或冻结测试成绩。','scope-note'));
    const grid=data.learning_rate_rope_grid||[],pairedRows=data.paired_seeds||[],parameterTables=el('div',null,'experiment-grid experiment-table-grid');
    if(grid.length)parameterTables.append(tablePanel('学习率对照',table([numeric('学习率'),numeric('RoPE 开 · 最佳 F1'),numeric('RoPE 关 · 最佳 F1'),numeric('配对差 / 百分点')],grid.map(r=>[String(r.encoder_lr),fmt(r.on),fmt(r.off),(r.delta_on_minus_off*100).toFixed(3)]))));
    parameterTables.append(tablePanel('三个种子的配对明细',table([numeric('种子'),numeric('开 · 最佳 F1'),numeric('关 · 最佳 F1'),numeric('最佳轮差 / 百分点'),numeric('第3轮差 / 百分点')],pairedRows.map(r=>[String(r.seed),fmt(r.on_best),fmt(r.off_best),(r.delta_best*100).toFixed(3),(r.delta_fixed3*100).toFixed(3)]))));
    panel.append(parameterTables);
    const stats=data.paired_statistics;
    if(stats?.delta_best){panel.append(el('p',`5e-5 · 三配对种子：最佳轮 F1 差 ${ (stats.delta_best.mean*100).toFixed(3)} ± ${(stats.delta_best.sample_sd*100).toFixed(3)} 个百分点；固定第3轮差 ${(stats.delta_fixed3.mean*100).toFixed(3)} ± ${(stats.delta_fixed3.sample_sd*100).toFixed(3)} 个百分点（样本标准差）。`,'study-stats'));}
    const training=data.training_curves||[],all=el('details',null,'development-details'),taskGrid=el('div',null,'experiment-grid experiment-table-grid');
    taskGrid.append(tablePanel('全部10组任务',table([numeric('学习率'),numeric('种子'),centered('RoPE'),numeric('最佳轮'),numeric('最佳开发 F1'),numeric('第3轮 F1')],training.map(r=>[String(r.lr),String(r.seed),r.rope?'开':'关',String(r.best_epoch),fmt(r.best),fmt(r.third)]))),figure('rope-paired-effects','完整学习率与种子配对分析图'));
    all.append(el('summary','全部任务与配对分析'),taskGrid);
    if(training.length){
      const browser=el('section',null,'epoch-browser'),filterHost=el('div',null,'epoch-filters'),filterSpecs=[['学习率','lr',value=>Number(value).toExponential(0)],['种子','seed',value=>String(value)]],recordHost=el('div',null,'epoch-record-grid'),pairIndex=new Map();
      training.forEach(row=>{const key=`${row.lr}|${row.seed}`;if(!pairIndex.has(key))pairIndex.set(key,{lr:row.lr,seed:row.seed,open:null,closed:null});const pair=pairIndex.get(key);pair[row.rope?'open':'closed']=row;});
      const pairs=[...pairIndex.values()],first=pairs.find(pair=>state.experimentSelection&&pair.lr===state.experimentSelection.lr&&pair.seed===state.experimentSelection.seed)||pairs[0];
      state.experimentSelection={lr:first.lr,seed:first.seed};
      function drawEpochRecord(){
        const selected=pairs.find(pair=>pair.lr===state.experimentSelection.lr&&pair.seed===state.experimentSelection.seed)||pairs[0];
        state.experimentSelection={lr:selected.lr,seed:selected.seed};filterHost.replaceChildren();
        filterSpecs.forEach(([label,key,format])=>{
          const valuesHost=el('div',null,'epoch-filter-group'),labelNode=el('span',label,'epoch-filter-label');valuesHost.append(labelNode);
          const matches=pairs.filter(pair=>filterSpecs.every(([,other])=>other===key||pair[other]===state.experimentSelection[other]));
          const values=[...new Set(matches.map(pair=>pair[key]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));
          values.forEach(value=>{const button=el('button',format(value),'epoch-filter-button');button.type='button';button.setAttribute('aria-pressed',String(value===state.experimentSelection[key]));button.onclick=()=>{
            let candidates=pairs.filter(pair=>pair[key]===value&&filterSpecs.every(([,other])=>other===key||pair[other]===state.experimentSelection[other]));
            if(!candidates.length)candidates=pairs.filter(pair=>pair[key]===value);
            const next=candidates[0];state.experimentSelection={lr:next.lr,seed:next.seed};drawEpochRecord();
          };valuesHost.append(button);});filterHost.append(valuesHost);
        });
        const opens=selected.open?.records||[],closed=selected.closed?.records||[],epochs=[...new Set([...opens,...closed].map(row=>row.epoch))].sort((a,b)=>a-b),openByEpoch=new Map(opens.map(row=>[row.epoch,row])),closedByEpoch=new Map(closed.map(row=>[row.epoch,row]));
        const metric=(record,key)=>record?key==='train_loss'?Number(record[key]).toFixed(4):fmt(record[key]):'—';
        const tableArea=tablePanel(`${Number(selected.lr).toExponential(0)} · 种子 ${selected.seed} · RoPE 开 / 关 · ${epochs.length} 轮`,table([numeric('轮次'),numeric('开 · 训练损失'),numeric('开 · 开发 F1'),numeric('关 · 训练损失'),numeric('关 · 开发 F1')],epochs.map(epoch=>[String(epoch),metric(openByEpoch.get(epoch),'train_loss'),metric(openByEpoch.get(epoch),'dev_f1'),metric(closedByEpoch.get(epoch),'train_loss'),metric(closedByEpoch.get(epoch),'dev_f1')])));
        const chartArea=el('section',null,'epoch-chart-stack');chartArea.append(el('h4','对应任务曲线'));
        const curveLabel=`${Number(selected.lr).toExponential(0)} · 种子 ${selected.seed} · RoPE 对照`,curves=[selected.open,selected.closed].filter(Boolean).map(row=>({...row,method:row.rope?'RoPE 开':'RoPE 关'}));
        [['训练损失','train_loss'],['开发集严格 F1','dev_f1']].forEach(([title,key])=>{const chart=el('section',null,'chart-area');chart.append(el('h5',title),plotCurves(curves,key,`${curveLabel} · ${title}`));chartArea.append(chart);});
        recordHost.replaceChildren(tableArea,chartArea);
      }
      browser.append(filterHost,recordHost);drawEpochRecord();panel.append(browser);
    }
    panel.append(all);
    const thresholdGrid=el('div',null,'experiment-grid experiment-table-grid'),thresholds=section('解码阈值敏感性');
    thresholdGrid.append(tablePanel('阈值实验数据',table(['方法',numeric('阈值'),numeric('精确率'),numeric('召回率'),numeric('严格 F1')],(data.threshold||[]).map(r=>[r.method,String(r.threshold),fmt(r.precision),fmt(r.recall),fmt(r.f1)]))),figure('threshold-tradeoff','阈值取舍图'));
    thresholds.append(thresholdGrid,el('p','正式默认阈值仍为0；此表只展示已保存的开发实验。','scope-note'));panel.append(thresholds);
  }
  function drawErrors(){
    const head=el('div',null,'experiment-panel-title');head.append(el('h3','N1 错误分层'),badge('共同500 · 冻结预测'));panel.append(head);
    const grid=el('div',null,'experiment-grid'),groups=section('遗漏实体分组'),bars=el('div',null,'error-bars'),values=data.error_groups||[],max=Math.max(1,...values.map(g=>g.count));
    values.forEach(g=>{const row=el('div',null,'error-bar-row'),track=el('div',null,'error-bar-track'),fill=el('div',null,'error-bar-fill');fill.style.width=`${g.count/max*100}%`;track.append(fill);row.append(el('span',g.label),track,el('span',g.count,'mono'));bars.append(row);});groups.append(bars);
    if(data.error_counts)groups.append(el('p',`参考 ${data.error_counts.gold} · TP ${data.error_counts.tp} · FP ${data.error_counts.fp} · FN ${data.error_counts.fn}`,'scope-note'));
    const stratified=section('嵌套与未登录实体');stratified.append(table(['范围',numeric('召回率')],[['嵌套实体',fmt(data.nested_recall)],['未登录实体（OOV）',fmt(data.oov_recall)]]));grid.append(groups,stratified);panel.append(grid);
    if(data.by_type){const byType=Object.entries(data.by_type).map(([key,value])=>({...value,label:names[key]||key})),typeGrid=el('div',null,'experiment-grid experiment-table-grid');
      typeGrid.append(tablePanel('九类实体数据',table(['类别',numeric('参考数'),numeric('精确率'),numeric('召回率'),numeric('F1')],Object.entries(data.by_type).map(([key,v])=>[names[key]||key,v.gold,fmt(v.precision),fmt(v.recall),fmt(v.f1)]))));
      const chartArea=el('section',null,'chart-area metric-chart-area'),chartHead=el('div',null,'metric-chart-head');chartHead.append(el('h4','各类精确率 · 召回率 · F1'));
      const legend=el('div',null,'metric-legend');[['precision','精确率'],['recall','召回率'],['f1','F1']].forEach(([key,label])=>{const item=el('span',null,'metric-legend-item');item.append(el('i',null,`metric-swatch metric-${key}`),el('span',label));legend.append(item);});
      chartArea.append(chartHead,legend,plotMetricBars(byType,'共同500条N1九类实体精确率、召回率与F1'));typeGrid.append(chartArea);panel.append(typeGrid);
    }
    panel.append(el('p','错误归组是输出观察，不是对模型内部机制的因果证明。','scope-note'));
  }
  function draw(tab){state.experimentTab=tab;buttons.forEach(([id,b])=>b.setAttribute('aria-pressed',String(id===tab)));panel.replaceChildren();({formal:drawFormal,development:drawDevelopment,errors:drawErrors}[tab]||drawFormal)();
    const source=el('details',null,'source-notes');source.append(el('summary','来源与完整口径'));if(state.project.generated_at)source.append(el('p',`展示摘要生成：${formatDate(state.project.generated_at)}`));(data.notes||[]).forEach(note=>source.append(el('p',note)));(data.sources||[]).forEach(path=>source.append(el('p',path,'mono')));panel.append(source);if(state.view==='experiments')reveal(panel);
  }
  [['formal','正式冻结评估'],['development','开发参数研究'],['errors','错误分层']].forEach(([id,label])=>{const b=el('button',label);b.type='button';b.onclick=()=>draw(id);buttons.push([id,b]);nav.append(b);});draw(state.experimentTab);
}
function renderAssets(){const assets=state.project?.assets||{},host=$('asset-list');host.replaceChildren();const titles={data:'官方数据',archive:'原始压缩包',D0:'D0 词典',N0:'N0 权重',N1:'N1 权重',L3:'三样本示例'};Object.entries(assets).forEach(([key,value])=>{const row=el('div',null,'asset-row');row.append(el('strong',titles[key]||key),el('span',value.ready?'已就绪':'缺失',`asset-status ${value.ready?'':'missing'}`),el('span',value.ready?'':value.message||''));host.append(row);});const needed=['data','N0','N1'];const missing=needed.filter(key=>assets[key]?.ready===false);$('repro-availability').textContent=missing.length?`待准备：${missing.map(k=>titles[k]).join('、')}。缺资产时入口将明确报告。`:'将创建新目录，不覆盖已有预测；同一时间只运行一个完整复现。';}
function renderHistory(){
  const history=$('repro-history');history.replaceChildren();
  const labels={passed:'通过',failed:'失败',cancelled:'已取消',PASS:'通过',PASS_WITH_NUMERICAL_DIFFERENCES:'存在数值差异',FAIL:'失败',UNKNOWN:'状态未记录'};
  const records=(state.savedJobs||[]).map(job=>({at:job.started_at,row:['本机会话完整复现',labels[job.status]||job.status||'状态未记录',formatDate(job.started_at),job.seconds==null?'—':seconds(job.seconds),job.output||'—']}));
  (state.project?.previous_reproductions||[]).forEach(record=>{
    const label=String(record.label||'已保存的 CPU 重评').replace('候选包解压后CPU重评','独立环境 CPU 重评');
    records.push({at:record.at,row:[label,labels[record.status]||record.status||'状态未记录',formatDate(record.at),record.seconds==null?'—':seconds(record.seconds),record.source||'—']});
  });
  $('repro-history-count').textContent=String(records.length);
  if(!records.length){history.append(el('p','尚无已保存的历史记录。','help'));return;}
  records.sort((left,right)=>String(right.at||'').localeCompare(String(left.at||'')));
  history.append(el('p','历史记录独立于本次运行，不代表本次通过。','help'),table(['记录来源',centered('状态'),'开始时间',numeric('耗时'),'输出目录 / 来源'],records.map(record=>record.row),'history-table'));
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
function renderReproductionSteps(job){
  const host=$('repro-steps'),p=job.progress&&typeof job.progress==='object'?job.progress:{},stage=String(job.stage||''),status=job.status||'running';
  const definitions=[
    ['校验模型资产','核对正式模型清单与文件校验值。'],
    ['准备并校验数据','检查固定数据划分与样本数量。'],
    ['D0 词典推理','生成 5000 条词典路线预测。'],
    ['N0 字符模型推理','在 CPU 上生成字符模型预测。'],
    ['N1 BERT 模型推理','在 CPU 上生成 BERT 路线预测。'],
    ['核对冻结指标','将本次新结果与冻结指标逐项核对。']
  ];
  const rawLog=Array.isArray(job.log)?job.log.join('\n'):typeof job.log==='string'?job.log:'',events=rawLog.split('\n').flatMap(line=>{try{const value=JSON.parse(line);return value&&typeof value==='object'?[value]:[];}catch{return[];}});
  const hasEvent=name=>events.some(event=>event.event===name),methodOrder={D0:0,N0:1,N1:2},methodIndex=methodOrder[p.method]??-1;
  const completedMethods=new Set(events.filter(event=>event.event==='method_complete'&&methodOrder[event.method]!==undefined).map(event=>event.method));
  if(methodIndex>=0&&p.frozen_metric_equal!==undefined)completedMethods.add(p.method);
  const summaryResults=(job.summary?.result||job.summary)?.results||{},summaryHasAllMethods=['D0','N0','N1'].every(method=>Object.prototype.hasOwnProperty.call(summaryResults,method));
  Object.keys(summaryResults).forEach(method=>{if(methodOrder[method]!==undefined)completedMethods.add(method);});
  const evaluationStarted=hasEvent('fresh_CPU_evaluation_start')||stage==='fresh_CPU_evaluation_start'||methodIndex>=0||summaryHasAllMethods;
  const modelCheckDone=hasEvent('model_assets_check')||evaluationStarted||hasEvent('data_verify');
  const dataCheckDone=hasEvent('data_verify')||evaluationStarted;
  const running=status==='running'||status==='starting',passed=status==='passed';
  let current=0;
  if(passed)current=5;
  else if(summaryHasAllMethods&&!running)current=5;
  else if(methodIndex>=0)current=2+methodIndex+(completedMethods.has(p.method)?1:0);
  else if(evaluationStarted||stage==='fresh_CPU_evaluation_start')current=2;
  else if(dataCheckDone)current=2;
  else if(modelCheckDone||stage.startsWith('data_')||hasEvent('data_prepared'))current=1;

  const stageNames=['assets','data','D0','N0','N1','metrics'];host.replaceChildren();
  definitions.forEach(([title,description],index)=>{
    const method=stageNames[index],isMethod=methodOrder[method]!==undefined;
    let done=index===0?modelCheckDone:index===1?dataCheckDone:isMethod?(completedMethods.has(method)||(methodIndex>methodOrder[method])):passed;
    if(passed)done=true;
    const phase=index===current&&!done?(running?'active':status==='cancelled'?'cancelled':'failed'):done?'done':'pending';
    const item=el('li',null,`repro-step is-${phase}`);if(phase==='active')item.setAttribute('aria-current','step');
    const top=el('div',null,'repro-step-top'),number=el('span',String(index+1),'repro-step-number'),stateLabel={done:'已完成',active:'进行中',pending:'等待中',failed:'未通过',cancelled:'已取消'}[phase];
    top.append(number,el('span',stateLabel,'repro-step-state'));
    item.append(top,el('h4',title),el('p',description));
    let percent=phase==='done'?100:phase==='pending'?0:null,detail=phase==='pending'?'等待上一步完成。':'正在处理。';
    if(phase==='done'){
      if(index===0)detail='模型文件与清单核验完成。';
      else if(index===1)detail='固定数据划分核验完成。';
      else if(isMethod){const completion=events.slice().reverse().find(event=>event.event==='method_complete'&&event.method===method),metric=completion||((p.method===method&&p.frozen_metric_equal!==undefined)?p:null);detail=metric?`严格 F1 ${fmt(metric.f1)} · 冻结核对 ${metric.frozen_metric_equal===true?'一致':metric.frozen_metric_equal===false?'不一致':'未取得'}`:'该路线已完成。';}
      else detail='本次复现与冻结指标核对通过。';
    }else if(phase==='active'){
      if(index===0)detail='正在核对模型文件与清单。';
      else if(index===1)detail='正在准备或核验固定数据。';
      else if(isMethod&&p.method===method&&Number(p.total_windows)>0){const complete=Math.min(Number(p.total_windows),Math.max(0,Number(p.completed_windows)||0));percent=Math.round(complete/Number(p.total_windows)*100);detail=`${complete} / ${Number(p.total_windows)} 个窗口 · ${percent}%`;}
      else if(method==='D0')detail='词典路线运行中；当前不提供窗口计数。';
      else if(isMethod)detail='CPU 推理启动中，等待窗口进度。';
      else detail='正在汇总结果并核对冻结指标。';
    }else if(phase==='failed')detail='当前步骤未通过；请查看错误信息和本次日志。';
    else if(phase==='cancelled')detail='任务已取消，部分输出已保留。';
    item.append(el('p',detail,'repro-step-detail'));
    const progress=document.createElement('progress');progress.max=100;progress.setAttribute('aria-label',`${title}进度`);if(percent===null)progress.removeAttribute('value');else progress.value=percent;item.append(progress);host.append(item);
  });
}
function renderJob(job){
  state.job=job;show('repro-job');show('repro-empty',false);const status=job.status||'running',running=status==='running'||status==='starting';
  $('repro-start').disabled=running;show('repro-cancel',running);$('repro-cancel').disabled=false;
  const labels={running:'运行中',starting:'启动中',passed:'本次复现通过',failed:'本次复现失败',cancelled:'本次已取消，部分输出保留'};
  $('repro-stage').textContent=`${state.restoredJobId===job.id?'恢复的任务 · ':''}${labels[status]||status} · ${job.stage||''}`;$('repro-time').textContent=seconds(job.seconds);
  $('repro-output').textContent=`输出：${job.output||'正在创建新目录'}　开始：${formatDate(job.started_at)}${job.finished_at?`　结束：${formatDate(job.finished_at)}`:''}`;
  const p=job.progress;let progress=typeof p==='number'?(p<=1?p*100:p):Number(p?.percent||0);
  if(p&&p.method){const order={D0:0,N0:1,N1:2},base=order[p.method]??0,ratio=p.total_windows?Math.min(1,Number(p.completed_windows||0)/Number(p.total_windows)):p.frozen_metric_equal!==undefined?1:0;progress=(base+ratio)/3*95;}
  if(status==='passed')progress=100;$('repro-progress').value=progress;renderReproductionSteps(job);
  const log=Array.isArray(job.log)?job.log.join('\n'):typeof job.log==='object'?JSON.stringify(job.log,null,2):job.log||'',logBox=$('repro-log'),nearBottom=logBox.scrollTop+logBox.clientHeight>=logBox.scrollHeight-35;
  logBox.textContent=log;if(nearBottom)logBox.scrollTop=logBox.scrollHeight;$('repro-summary').replaceChildren();
  if(job.summary){const summary=summaryDTO(job.summary);$('repro-summary').append(reproductionTable(summary));const details=el('details');details.append(el('summary','原始核对摘要 JSON'),el('pre',JSON.stringify(summary,null,2),'json-box'));$('repro-summary').append(details);}
  $('repro-error').textContent=job.error?(typeof job.error==='string'?job.error:JSON.stringify(job.error)):'';
  if(running)scheduleJobPoll(job.id);else{if(state.poll){clearTimeout(state.poll);state.poll=null;}state.savedJobs=(state.savedJobs||[]).filter(item=>item.id!==job.id);state.savedJobs.push(job);renderHistory();}
  announce(`${labels[status]||status}。`,'delivery');
}
function scheduleJobPoll(id){if(state.poll)clearTimeout(state.poll);state.poll=setTimeout(async()=>{try{renderJob(await request(`/api/reproduction/jobs/${encodeURIComponent(id)}`));}catch(err){$('repro-error').textContent=`任务状态暂未读取：${safeError(err)}；正在重试。`;scheduleJobPoll(id);}},1500);}
async function restoreJobs(){
  try{const list=await request('/api/reproduction/jobs'),jobs=Array.isArray(list)?list:list.jobs||[];
    if(list.protection?.locked)$('repro-error').textContent=list.protection.message||'发现复现保护锁，需先核查原任务。';
    const active=list.active||jobs.find(job=>job.status==='running');state.savedJobs=jobs.filter(job=>job.id!==active?.id);renderHistory();
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
  if(view==='experiments')renderExperiments();
  if(view==='compare'&&!state.compare.result&&!state.compare.busy){$('compare-text').value=$('live-text').value;updateCounts();}
  if(state.compare.busy&&view!=='compare')invalidateCompare();
  $('footer-status').textContent=state.messages[view]||'';updateRuntime();$('main').focus({preventScroll:true});reveal($('view-'+view));
}
let experimentResizeTimer;window.addEventListener('resize',()=>{clearTimeout(experimentResizeTimer);experimentResizeTimer=setTimeout(()=>{if(state.view==='experiments')renderExperiments();},150);});
document.querySelectorAll('.main-nav button').forEach(button=>button.onclick=()=>navigate(button.dataset.view));$('open-api').onclick=openApi;$('close-api').onclick=closeApi;$('api-dialog').addEventListener('close',()=>{$('open-api').focus();});$('api-dialog').addEventListener('click',event=>{if(event.target===$('api-dialog')){const rect=$('api-dialog').getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)closeApi();}});$('api-form').onsubmit=checkApi;$('api-test').onclick=checkModels;$('api-provider').addEventListener('change',applyProvider);$('api-key-toggle').onclick=()=>{const visible=$('api-key').type==='password';$('api-key').type=visible?'text':'password';$('api-key-toggle').setAttribute('aria-pressed',String(visible));$('api-key-toggle').setAttribute('aria-label',visible?'隐藏 API 密钥':'显示 API 密钥');};$('api-base').addEventListener('input',changeAddress);$('api-model').addEventListener('change',changeApiModel);$('api-model-manual-toggle').onclick=()=>{const before=currentApiModelId();setManualModelMode(!state.apiModelManual);if(before!==currentApiModelId())changeApiModel();};$('api-model-manual').addEventListener('input',changeManualModel);$('api-form').addEventListener('invalid',event=>{focusInvalid(event.target,'请检查此字段。');},true);['api-key'].forEach(id=>$(id).addEventListener('input',event=>{event.target.removeAttribute('aria-invalid');persistApiPreferences();uncheckApi();}));$('api-clear-key').onclick=()=>{$('api-key').value='';hideApiKey();uncheckApi();announce('API 密钥已从当前页面清除。');};$('live-run').onclick=runLive;$('live-text').oninput=()=>{invalidateLive();show('sample-purpose',false);$('live-samples').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed','false'));};$('live-method').onchange=()=>{invalidateLive();updateMethodNotice();};$('live-clear').onclick=()=>{show('sample-purpose',false);$('live-text').value='';invalidateLive('已清空原文，请输入待抽取文本。');$('live-text').focus();};$('live-text').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();runLive();}});$('live-json').onclick=()=>{if(!state.live.result)return;setLiveJsonVisible($('live-json-details').hidden);};$('live-copy').onclick=copyJson;$('live-export').onclick=()=>exportJson(state.live.result,`医学实体_${state.live.result?.method||'结果'}_现场.json`);$('compare-text').oninput=invalidateCompare;document.querySelectorAll('.compare-method').forEach(input=>input.onchange=invalidateCompare);$('compare-run').onclick=runCompare;$('compare-export').onclick=()=>exportJson(state.compare.result,'医学实体_当前方法对比.json');$('case-select').onchange=loadCase;$('case-route').onchange=renderCase;$('case-export').onclick=()=>exportJson(state.cases.result,`医学实体_${state.cases.result?.selected_case}_${state.cases.result?.selected_route}_历史证据.json`);$('refresh-project').onclick=refreshProject;$('repro-start').onclick=startRepro;$('repro-cancel').onclick=cancelRepro;

$('case-only-diff').onchange=renderCase;
$('repro-history-open').onclick=()=>{renderHistory();$('repro-history-dialog').showModal();};
$('close-repro-history').onclick=()=>$('repro-history-dialog').close();
$('repro-history-dialog').addEventListener('close',()=>{$('repro-history-open').focus();});
$('live-help-toggle').onclick=()=>{const expanded=$('live-guide').hidden;show('live-guide',expanded);$('live-help-toggle').setAttribute('aria-expanded',String(expanded));};
$('live-expand').onclick=()=>{const expanded=$('live-expand').getAttribute('aria-expanded')!=='true';$('live-expand').setAttribute('aria-expanded',String(expanded));$('live-expand').textContent=expanded?'收起输入':'展开输入';if(expanded)$('live-text').classList.add('expanded');else $('live-text').classList.remove('expanded');};
document.addEventListener?.('keydown',()=>{state.keyboard=true;document.body.setAttribute('data-keyboard','true');},true);
document.addEventListener?.('pointerdown',()=>{state.keyboard=false;document.body.setAttribute('data-keyboard','false');},true);

window.addEventListener('pagehide',()=>{$('api-key').value='';hideApiKey();uncheckApi();});
updateApiLayout();updateCounts();refreshProject();loadCases();loadApiPresets();globalThis.TeacherSelect?.refresh();
})();
