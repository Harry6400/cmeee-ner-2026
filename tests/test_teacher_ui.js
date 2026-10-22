'use strict';
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const root=process.argv[2]||path.resolve(__dirname,'..');

async function main(){
  const source=fs.readFileSync(root+'/ui/teacher.js','utf8');
  const html=fs.readFileSync(root+'/ui/stage5.html','utf8');
  class Classes {constructor(){this.set=new Set();}add(...x){x.forEach(v=>this.set.add(v));}remove(...x){x.forEach(v=>this.set.delete(v));}contains(v){return this.set.has(v);}}
  class Element {
    constructor(tag='div'){this.tagName=tag;this.children=[];this.attrs={};this.listeners={};this.classList=new Classes();this.dataset={};this.value='';this.disabled=false;this.hidden=false;this.checked=false;this.open=false;this._text='';this.style={};}
    set textContent(v){this._text=String(v??'');this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent??String(x)).join('');}
    set className(v){this._class=String(v);this.classList=new Classes();this.classList.add(...String(v).split(' '));}get className(){return this._class||'';}
    append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this._text='';this.children=[...nodes];}
    setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]??null;}removeAttribute(k){delete this.attrs[k];}
    addEventListener(k,f){(this.listeners[k]??=[]).push(f);}dispatch(k){for(const f of this.listeners[k]||[])f({target:this});}
    focus(){}scrollIntoView(){}showModal(){this.open=true;}close(){this.open=false;(this.listeners.close||[]).forEach(f=>f());}
    querySelectorAll(selector){if(this.id==='api-form'&&selector==='input,select')return apiInputs;return this.children.flatMap(c=>[c,...(c.querySelectorAll?.(selector)||[])]).filter(c=>selector==='button'?c.tagName==='button':selector==='.active'?c.classList?.contains('active'):selector==='.data-table'?c.classList?.contains('data-table'):false);}
    querySelector(){return null;}remove(){}click(){this.onclick?.({preventDefault(){}});}getBoundingClientRect(){return {left:0,top:0,right:100,bottom:100};}
  }
  const elements=new Map([...html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"/g)].map(m=>{const n=new Element(m[1]);n.id=m[2];return[m[2],n];}));
  const $=id=>elements.get(id);
  const apiInputs=['api-provider','api-model','api-base','api-key','api-model-manual'].map($);
  Object.entries({'api-provider':'deepseek','live-text':'出现发热。','live-method':'N1','compare-text':'出现发热。'}).forEach(([id,v])=>$(id).value=v);
  $('api-key').type='password';
  const nav=['live','compare','errors','experiments','delivery'].map(view=>{const e=new Element('button');e.dataset.view=view;return e;});
  const checks=['D0','N0','N1','L0','L3'].map(method=>{const e=new Element('input');e.value=method;e.checked=method==='L0';return e;});
  const calls=[],stored=new Map([['teacher-api-preferences-v1',JSON.stringify({provider:'openrouter',model:'openai/gpt-4o-mini',input_price_cny:'legacy-price'})]]),localStorage={getItem:key=>stored.get(key)||null,setItem:(key,value)=>stored.set(key,String(value)),removeItem:key=>stored.delete(key)};
  const context={document:{getElementById:$,createElement:tag=>new Element(tag),createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e;},createElementNS:(_ns,tag)=>new Element(tag),body:new Element('body'),addEventListener(){},querySelectorAll:selector=>selector==='.main-nav button'?nav:selector==='.compare-method'?checks:selector==='.compare-method:checked'?checks.filter(c=>c.checked):[]},Node:Element,Option:class extends Element{constructor(text,value){super('option');this.textContent=text;this.value=value;}},performance,crypto:globalThis.crypto,fetch:(url,options)=>new Promise(resolve=>calls.push({url,options,resolve})),setInterval:()=>1,clearInterval(){},setTimeout:()=>1,clearTimeout(){},console,Blob,URL,navigator:{clipboard:{writeText:async()=>{}}},localStorage,window:{addEventListener(){}},TeacherSelect:{refresh(){},close(){},focus(){}},globalThis:null};context.globalThis=context;
  vm.runInNewContext(source,context,{filename:'ui/teacher.js'});
  function pending(url){const call=[...calls].reverse().find(c=>c.url===url&&!c.responded);assert.ok(call,`Missing expected mocked request ${url}`);return call;}
  function respond(call,data){call.responded=true;call.resolve({ok:true,status:200,json:async()=>data});}
  function flatten(node){return [node,...(node?.children||[]).flatMap(flatten)];}
  function findClass(node,className){return flatten(node).find(item=>item.classList?.contains(className));}
  function models(ids,prices={}){return ids.map(id=>({id,label:id,json_mode:true,...(prices[id]||{})}));}
  const priced={input_price_cny:'2',output_price_cny:'8',context_max_tokens:1048576,price_source:'https://example.org/pricing',price_checked_at:'2026-10-21',price_is_reference:true};
  const deepseekModels=models(['deepseek-flash','deepseek-v4-pro'],{'deepseek-flash':{...priced},'deepseek-v4-pro':{...priced}});
  const providerSpecs=[
    ['deepseek','DeepSeek','openai_compatible','https://api.deepseek.com','deepseek-flash',deepseekModels],
    ['openai','OpenAI','openai_compatible','https://api.openai.com/v1','gpt-6.1-sol',models(['gpt-6.1-sol'])],
    ['anthropic','Anthropic','anthropic','https://api.anthropic.com','claude-sonnet-5-5',models(['claude-sonnet-5-5'],{'claude-sonnet-5-5':{...priced}})],
    ['gemini','Gemini API','openai_compatible','https://generativelanguage.googleapis.com/v1beta/openai','gemini-3.8-flash',models(['gemini-3.8-flash'],{'gemini-3.8-flash':{...priced}})],
    ['dashscope','阿里云百炼 / Qwen','openai_compatible','https://dashscope.aliyuncs.com/compatible-mode/v1','qwen3.8-flash',models(['qwen3.8-flash'],{'qwen3.8-flash':{...priced}})],
    ['moonshot','月之暗面 / Kimi','openai_compatible','https://api.moonshot.cn/v1','kimi-k3',models(['kimi-k3'],{'kimi-k3':{...priced}})],
    ['custom','OpenAI 兼容自定义','openai_compatible','','',[]]
  ];
  const providers=providerSpecs.map(([id,label,protocol,base_url,default_model,items])=>({id,label,protocol,base_url,default_model,model_path:id==='anthropic'?'/v1/models':id==='gemini'?'/v1beta/models':'/models',model_discovery:true,models:items}));
  const training=[];for(const [i,[lr,seed]] of [[1e-5,20261019],[3e-5,20261019],[5e-5,20261019],[5e-5,20261020],[5e-5,20261021]].entries())for(const rope of [true,false])training.push({lr,seed,rope,best_epoch:2,best:.31+i*.01,third:.29+i*.01,records:[{epoch:1,train_loss:1+i*.1,dev_f1:.25+i*.01},{epoch:2,train_loss:.7+i*.1,dev_f1:.31+i*.01}]});
  const experimentFixture={training_curves:training,learning_rate_rope_grid:[],paired_seeds:[],threshold:[]};
  const tick=()=>new Promise(resolve=>setImmediate(resolve));
  const checksResult=[];

  respond(pending('/api/api-presets'),{provider:'deepseek',default_model:'deepseek-flash',base_url:'https://api.deepseek.com',models:deepseekModels,providers,source:'local_registry',external_requests:0});
  respond(pending('/api/cases'),[]);
  respond(pending('/api/project'),{title:'中文医学实体抽取',people:[{name:'吴皓睿',student_id:'示例编号甲'},{name:'周栩乐',student_id:'示例编号乙'}],assets:{},experiments:experimentFixture,artifacts:[],previous_reproductions:[{label:'阶段六初版：候选包解压后CPU重评',at:'2026-10-20T00:00:00Z',status:'PASS',seconds:2,source:'synthetic-history-fixture'}],links:{}});
  await tick();

  assert.equal($('identity').textContent,'吴皓睿 示例编号甲　/　周栩乐 示例编号乙');
  assert.doesNotMatch(html,/api-advanced|高级设置|api-json/);
  assert.equal($('api-model').children.length,3);
  assert.equal($('api-provider').value,'deepseek');
  assert.equal($('api-provider').children.length,7);
  assert.equal($('api-key').type,'password');
  assert.equal($('api-provider').tagName,'select');
  assert.equal($('api-model').tagName,'select');
  for(const provider of providers){
    $('api-provider').value=provider.id;$('api-provider').dispatch('change');
    assert.equal($('api-provider').value,provider.id);
    const known=provider.models;
    assert.equal($('api-model').value,known[0]?.id||'');
    for(const option of $('api-model').children.filter(item=>item.value))assert.notEqual(option.disabled,true);
  }
  $('api-provider').value='deepseek';$('api-provider').dispatch('change');
  assert.equal($('repro-history-dialog').open,false);
  $('repro-history-open').click();
  assert.equal($('repro-history-dialog').open,true);
  assert.ok(findClass($('repro-history'),'history-table'));
  assert.match($('repro-history').textContent,/开始时间/);
  assert.match(html,/class="acceptance-guide"[\s\S]*?老师验收路径[\s\S]*?<\/section>/);
  assert.match(html,/class="command-reference"[\s\S]*?<pre class="command"/);
  assert.doesNotMatch(html,/提交材料|录屏与独立资源|复现与提交|本地候选/);
  checksResult.push('DeepSeek defaults, hidden history dialog with a table, visible acceptance route, visible command reference');

  // Provider and model are dropdowns; prices resolve from the local model directory.
  $('api-provider').value='anthropic';$('api-provider').dispatch('change');
  assert.equal($('api-provider').value,'anthropic');
  assert.equal($('api-model').value,'claude-sonnet-5-5');
  assert.equal($('api-key').value,'');
  assert.match($('api-price-summary').textContent,/实际费用以服务商账单为准/);
  $('api-key-toggle').click();assert.equal($('api-key').type,'text');assert.equal($('api-key-toggle').getAttribute('aria-pressed'),'true');
  $('api-key-toggle').click();assert.equal($('api-key').type,'password');assert.equal($('api-key-toggle').getAttribute('aria-pressed'),'false');
  $('api-key').value='private-session-only';$('api-key').dispatch('input');
  assert.doesNotMatch([...stored.values()].join('\n'),/private-session-only/);
  checksResult.push('All provider dropdown selections, unpriced models selectable, accessible key reveal toggle, memory-only API key');

  // The experimental browser pairs both RoPE variants and follows all five saved LR/seed combinations.
  nav.find(button=>button.dataset.view==='experiments').onclick();
  findClass($('experiments-content'),'experiment-tabs').children[1].click();
  const browser=findClass($('experiments-content'),'epoch-browser');assert.ok(browser);
  const filters=findClass(browser,'epoch-filters'),records=findClass(browser,'epoch-record-grid');
  function choose(index,label){const group=filters.children[index],button=group.children.find(node=>node.tagName==='button'&&node.textContent===label);assert.ok(button,`Missing filter option ${label}`);button.click();}
  const pairs=[];for(const row of training){const key=`${row.lr}|${row.seed}`;if(!pairs.some(pair=>pair.key===key))pairs.push({key,lr:row.lr,seed:row.seed});}
  for(const pair of pairs){
    choose(0,Number(pair.lr).toExponential(0));choose(1,String(pair.seed));
    const selected=findClass(records,'experiment-table-panel').children[0].textContent;
    assert.match(selected,new RegExp(Number(pair.lr).toExponential(0).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
    assert.match(selected,new RegExp(String(pair.seed)));
    assert.match(selected,/RoPE 开 \/ 关/);
    const chartLabels=flatten(records).filter(node=>node.tagName==='svg').map(node=>node.getAttribute('aria-label')||'');
    assert.equal(chartLabels.length,2);assert.ok(chartLabels.every(label=>label.includes(Number(pair.lr).toExponential(0))&&label.includes(String(pair.seed))&&label.includes('RoPE 对照')));
    for(const svg of flatten(records).filter(node=>node.tagName==='svg')){assert.equal(flatten(svg).filter(node=>node.tagName==='polyline').length,2);assert.match(svg.textContent,/RoPE 开/);assert.match(svg.textContent,/RoPE 关/);}
    assert.equal(findClass(browser,'data-table')!==undefined,true);
    assert.equal(flatten(browser).filter(node=>node.classList?.contains('data-table')).length,1);
  }
  checksResult.push('One paired epoch table and two stacked loss/F1 charts across all five saved combinations');

  // Obsolete local-check responses cannot bless settings edited in flight.
  $('api-provider').value='deepseek';$('api-provider').dispatch('change');
  $('api-key').value='race-session-key';
  $('api-model').value='deepseek-v4-pro';$('api-model').dispatch('change');
  const first=$('api-form').onsubmit({preventDefault(){}}),firstCall=pending('/api/api-check');
  $('api-model').value='deepseek-flash';$('api-model').dispatch('change');
  respond(firstCall,{status:'checked_locally',endpoint:'https://api.deepseek.com/chat/completions',reservation_cny:'2.129920'});await first;
  assert.equal($('api-config-dot').classList.contains('ready'),false);assert.match($('api-form-error').textContent,/设置已改变/);
  checksResult.push('Local API-check race discards outdated settings');

  // Editing live text while local inference is in flight invalidates its stale result.
  const local=$('live-run').onclick(),localCall=pending('/api/extract');$('live-text').value='修改后的原文。';$('live-text').oninput();
  respond(localCall,{method:'N1',prediction:[],status:'valid',seconds:1,source:'teacher_live_cpu'});await local;
  assert.equal($('live-export').disabled,true);assert.match($('live-result').textContent,/重新运行/);
  checksResult.push('Live inference race invalidates results after source text edits');

  // Applying configuration is local; known prices come from the directory, not browser-entered rates.
  $('api-key').value='not-a-real-secret';const stable=$('api-form').onsubmit({preventDefault(){}}),stableCall=pending('/api/api-check');
  const stableConfig=JSON.parse(stableCall.options.body).config;
  assert.equal(stableConfig.model,'deepseek-flash');assert.equal(stableConfig.json_mode,true);assert.equal(Object.hasOwn(stableConfig,'input_price_cny'),false);assert.equal(Object.hasOwn(stableConfig,'output_price_cny'),false);
  respond(stableCall,{status:'checked_locally',endpoint:'https://api.deepseek.com/chat/completions',reservation_cny:'2.129920'});await stable;
  assert.equal($('api-config-dot').classList.contains('ready'),true);
  assert.ok(calls.every(call=>call.url.startsWith('/api/')));
  checksResult.push('Saving uses no price fields and performs no external fetch');

  $('live-method').value='L0';$('live-method').onchange();
  const liveApi=$('live-run').onclick(),liveApiCall=pending('/api/extract-api');
  assert.equal($('live-run').disabled,true);await $('live-run').onclick();
  respond(liveApiCall,{method:'L0',status:'valid',prediction:[],source:'teacher_live_api',model:'deepseek-flash',cost:{state:'unknown',estimated_cny:null},usage:null,seconds:.1});await liveApi;
  assert.match($('live-result').textContent,/参考费用：未知/);assert.match($('live-json-content').textContent,/"estimated_cny": null/);
  assert.doesNotMatch($('live-json-content').textContent,/not-a-real-secret/);
  checksResult.push('API extraction preserves unknown cost as null, displays unknown, and blocks duplicate clicks');

  // A custom model is usable immediately without a pricing form or directory refresh.
  $('api-provider').value='custom';$('api-provider').dispatch('change');
  assert.equal($('api-provider').value,'custom');assert.equal($('api-key').value,'');
  assert.equal($('api-model-manual-wrap').hidden,false);
  assert.doesNotMatch(html,/api-custom-pricing|api-input-price|api-context|api-price-source/);
  $('api-base').value='https://example.org/v1';$('api-base').dispatch('input');$('api-key').value='temporary-custom-key';
  $('api-model-manual').value='course-model';$('api-model-manual').dispatch('input');
  const generationsBeforeListing=calls.filter(call=>call.url==='/api/extract-api').length;
  const manual=$('api-form').onsubmit({preventDefault(){}}),manualCall=pending('/api/api-check'),manualConfig=JSON.parse(manualCall.options.body).config;
  assert.equal(manualConfig.model,'course-model');assert.equal(Object.hasOwn(manualConfig,'input_price_cny'),false);
  respond(manualCall,{status:'checked_locally',endpoint:'https://example.org/v1/chat/completions'});await manual;
  assert.equal($('api-dialog').open,false);assert.equal($('api-config-dot').classList.contains('ready'),true);
  const unsupportedCustom=$('api-test').onclick();respond(pending('/api/api-models'),{status:'models_unsupported',selected_model_found:null,models:[],generation_requests:0});await unsupportedCustom;
  assert.equal($('api-model-manual').value,'course-model');assert.equal($('api-config-dot').classList.contains('ready'),true);
  assert.equal(calls.filter(call=>call.url==='/api/extract-api').length,generationsBeforeListing);
  assert.doesNotMatch([...stored.values()].join('\n'),/temporary-custom-key|input_price_cny|output_price_cny|price_source|context_max_tokens/);
  checksResult.push('Custom model saves directly with URL and key, no price form; legacy preferences cleaned; no implicit requests');

  // Switch to DeepSeek Pro; model-list refreshes query the directory only.
  $('api-provider').value='deepseek';$('api-provider').dispatch('change');
  $('api-key').value='not-a-real-secret';$('api-model').value='deepseek-v4-pro';$('api-model').dispatch('change');
  const pro=$('api-form').onsubmit({preventDefault(){}}),proCall=pending('/api/api-check'),proConfig=JSON.parse(proCall.options.body).config;
  assert.equal(proConfig.model,'deepseek-v4-pro');assert.equal(Object.hasOwn(proConfig,'input_price_cny'),false);assert.equal(Object.hasOwn(proConfig,'output_price_cny'),false);
  respond(proCall,{status:'checked_locally',endpoint:'https://api.deepseek.com/chat/completions'});await pro;
  const generations=calls.filter(call=>call.url==='/api/extract-api').length;
  const connection=$('api-test').onclick(),listRequest=pending('/api/api-models'),listConfig=JSON.parse(listRequest.options.body).config;
  assert.equal(Object.hasOwn(listConfig,'model'),false);respond(listRequest,{status:'models_checked',selected_model_found:null,models:[{...deepseekModels[1]}],generation_requests:0});await connection;
  assert.match($('api-validation').textContent,/未发送待抽取文本/);assert.equal(calls.filter(call=>call.url==='/api/extract-api').length,generations);
  const fresh=$('api-test').onclick();respond(pending('/api/api-models'),{status:'models_checked',models:[{id:'new-unpriced-model',label:'New model'}]});await fresh;
  const newOption=$('api-model').children.find(option=>option.value==='new-unpriced-model');assert.ok(newOption);assert.notEqual(newOption.disabled,true);
  assert.equal($('api-model').value,'deepseek-v4-pro');
  const failed=$('api-test').onclick(),failedCall=pending('/api/api-models');failedCall.responded=true;failedCall.resolve({ok:false,status:401,json:async()=>({error:'密钥未通过服务鉴权'})});await failed;
  assert.ok($('api-model').children.some(option=>option.value==='deepseek-v4-pro'));assert.equal($('api-config-dot').classList.contains('ready'),true);
  checksResult.push('Fresh unpriced models remain enabled and failed model refresh preserves the saved selection');
  const unsupported=$('api-test').onclick();respond(pending('/api/api-models'),{status:'models_unsupported',selected_model_found:null,models:deepseekModels,generation_requests:0});await unsupported;
  assert.match($('api-validation').textContent,/未提供模型目录/);assert.equal($('api-config-dot').classList.contains('ready'),true);
  const changing=$('api-test').onclick(),directory=pending('/api/api-models');$('api-model').value='deepseek-flash';$('api-model').dispatch('change');respond(directory,{status:'models_checked',selected_model_found:null,models:deepseekModels,generation_requests:0});await changing;
  assert.equal($('api-config-dot').classList.contains('ready'),false);assert.match($('api-form-error').textContent,/结果已丢弃/);
  assert.ok(calls.filter(call=>call.url==='/api/extract-api').every(call=>JSON.parse(call.options.body).config.api_key==='not-a-real-secret'));
  checksResult.push('Explicit model refresh checks availability without extraction and discards stale directory results');

  console.log(JSON.stringify({status:'PASS',harness:'Offline Node DOM state harness; synthetic data and HTTP responses only',external_requests:0,checks:checksResult},null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
