'use strict';
// Shared select-only combobox. Native selects retain values/events, never open OS menus.
(() => {
  const controllers=new Set(),mounted=new WeakMap();let opened=null,serial=0,queued=false;
  function mount(select){
    if(mounted.has(select))return;
    const wrap=document.createElement('div'),trigger=document.createElement('button'),value=document.createElement('span'),arrow=document.createElement('span'),list=document.createElement('div');
    wrap.className='choice-select';trigger.className='choice-trigger';trigger.type='button';trigger.setAttribute('role','combobox');trigger.setAttribute('aria-haspopup','listbox');trigger.setAttribute('aria-expanded','false');
    value.className='choice-value';arrow.className='choice-chevron';arrow.setAttribute('aria-hidden','true');trigger.append(value,arrow);
    list.className='choice-menu';list.id=`choice-list-${++serial}`;list.setAttribute('role','listbox');list.setAttribute('popover','manual');list.hidden=true;trigger.setAttribute('aria-controls',list.id);
    select.after(wrap);wrap.append(select,trigger,list);select.hidden=true;select.tabIndex=-1;select.setAttribute('aria-hidden','true');select.dataset.customSelect='true';
    let signature='',options=[],items=[],active=-1,typed='',typedAt=0;
    const enabled=()=>options.map((o,i)=>!o.disabled?i:-1).filter(i=>i>=0);
    function label(){return select.getAttribute('aria-label')||select.labels?.[0]?.textContent?.trim()||'选择选项';}
    function paint(){items.forEach((item,i)=>{item.classList.toggle('is-active',i===active);item.setAttribute('aria-selected',String(options[i].value===select.value));});if(active>=0)trigger.setAttribute('aria-activedescendant',items[active].id);}
    function move(index){active=index;paint();items[active]?.scrollIntoView({block:'nearest'});}
    function position(){
      if(opened!==api)return;const r=trigger.getBoundingClientRect(),gap=6,vw=document.documentElement.clientWidth,vh=window.innerHeight;
      const below=vh-r.bottom-gap-10,above=r.top-gap-10,up=below<180&&above>below;
      list.style.width=`${Math.min(Math.max(r.width,230),vw-20)}px`;list.style.maxHeight=`${Math.max(40,Math.min(300,up?above:below))}px`;
      const height=list.getBoundingClientRect().height;list.style.left=`${Math.max(10,Math.min(r.left,vw-list.getBoundingClientRect().width-10))}px`;list.style.top=`${Math.max(10,up?r.top-gap-height:r.bottom+gap)}px`;
    }
    function close(focus=false){if(opened!==api)return;opened=null;trigger.setAttribute('aria-expanded','false');trigger.removeAttribute('aria-activedescendant');if(list.matches(':popover-open'))list.hidePopover();list.hidden=true;if(focus&&trigger.isConnected)trigger.focus({preventScroll:true});}
    function open(index){
      if(select.disabled||!enabled().length)return;opened?.close();sync();opened=api;list.hidden=false;if(list.showPopover)list.showPopover();trigger.setAttribute('aria-expanded','true');active=index??options.findIndex(o=>o.value===select.value&&!o.disabled);if(active<0)active=enabled()[0];position();paint();items[active]?.scrollIntoView({block:'nearest'});
    }
    function choose(index){const option=options[index];if(!option||option.disabled)return;const changed=select.value!==option.value;select.value=option.value;sync();close(true);if(changed){select.dispatchEvent(new Event('input',{bubbles:true}));select.dispatchEvent(new Event('change',{bubbles:true}));}}
    function sync(){
      options=Array.from(select.options);const selected=options.find(o=>o.value===select.value),text=selected?.textContent||'请选择';if(value.textContent!==text)value.textContent=text;
      if(trigger.disabled!==select.disabled)trigger.disabled=select.disabled;trigger.setAttribute('aria-label',label());list.setAttribute('aria-label',label());
      for(const attr of ['aria-invalid','aria-describedby','aria-required']){const next=select.getAttribute(attr);if(next!==null){if(trigger.getAttribute(attr)!==next)trigger.setAttribute(attr,next);}else trigger.removeAttribute(attr);}
      const next=JSON.stringify(options.map(o=>[o.value,o.textContent,o.disabled]));
      if(signature!==next){signature=next;list.replaceChildren();items=options.map((option,i)=>{const item=document.createElement('div'),text=document.createElement('span'),check=document.createElement('span');item.className='choice-option';item.id=`${list.id}-option-${i}`;item.setAttribute('role','option');item.setAttribute('aria-disabled',String(option.disabled));text.className='choice-option-text';text.textContent=option.textContent;
          if(select.id==='api-model'){const detail=document.createElement('small');detail.textContent=option.value;text.append(detail);}check.className='choice-check';check.textContent='✓';check.setAttribute('aria-hidden','true');item.append(text,check);item.addEventListener('pointerdown',event=>event.preventDefault());item.addEventListener('click',()=>choose(i));item.addEventListener('pointermove',()=>{if(!option.disabled&&active!==i){active=i;paint();}});list.append(item);return item;});}
      if(opened===api){if(select.disabled)close();else paint();}
      else items.forEach((item,i)=>{const selected=String(options[i].value===select.value);if(item.getAttribute('aria-selected')!==selected)item.setAttribute('aria-selected',selected);});
    }
    const api={select,trigger,wrap,list,sync,close,position};mounted.set(select,api);controllers.add(api);sync();
    trigger.addEventListener('click',()=>opened===api?close():open());
    trigger.addEventListener('keydown',event=>{
      const key=event.key,indices=enabled();if(!indices.length)return;
      if(key==='Tab'){close();return;}
      if(key==='Escape'&&opened===api){event.preventDefault();event.stopPropagation();close(true);return;}
      if(['ArrowDown','ArrowUp','Home','End'].includes(key)){event.preventDefault();if(opened!==api)open();const at=indices.indexOf(active);move(key==='Home'?indices[0]:key==='End'?indices.at(-1):indices[Math.max(0,Math.min(indices.length-1,at+(key==='ArrowDown'?1:-1)))]);return;}
      if(key==='Enter'||key===' '){event.preventDefault();opened===api?choose(active):open();return;}
      if(key.length===1&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();const time=Date.now();typed=time-typedAt>700?'':typed;typed+=key.toLocaleLowerCase();typedAt=time;if(opened!==api)open();const query=[...typed].every(c=>c===typed[0])?typed[0]:typed;
        const ordered=[...indices.filter(i=>i>active),...indices.filter(i=>i<=active)],found=ordered.find(i=>options[i].textContent.trim().toLocaleLowerCase().startsWith(query));if(found!==undefined)move(found);}
    });
    select.addEventListener('change',sync);select.addEventListener('input',sync);
  }
  function refresh(){
    document.querySelectorAll('select').forEach(mount);
    controllers.forEach(api=>{if(!api.select.isConnected){api.close();controllers.delete(api);}else api.sync();});
  }
  new MutationObserver(()=>{if(!queued){queued=true;queueMicrotask(()=>{queued=false;refresh();});}}).observe(document.documentElement,{subtree:true,childList:true,attributes:true,attributeFilter:['disabled','aria-invalid','aria-describedby','label','value','selected']});
  document.addEventListener('pointerdown',event=>{if(opened&&!opened.wrap.contains(event.target))opened.close();},true);
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&opened){event.preventDefault();event.stopImmediatePropagation();opened.close(true);}},true);
  document.addEventListener('click',event=>{const label=event.target.closest('label[for]'),select=label&&document.getElementById(label.htmlFor),api=select&&mounted.get(select);if(api){event.preventDefault();api.trigger.focus();}});
  window.addEventListener('resize',()=>opened?.position());window.addEventListener('scroll',event=>{if(opened&&!opened.list.contains(event.target))opened.position();},true);
  document.addEventListener('toggle',event=>{if(event.target.tagName==='DIALOG'&&!event.target.open)opened?.close();},true);
  globalThis.TeacherSelect={refresh,close:()=>opened?.close(),focus:select=>{refresh();mounted.get(typeof select==='string'?document.getElementById(select):select)?.trigger.focus();}};
  refresh();
})();
