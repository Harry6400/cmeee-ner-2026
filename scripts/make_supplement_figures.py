"""Template-aligned system diagrams and real processed-dev threshold results (Python)."""
from pathlib import Path
import csv,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/template-revision/figures';OUT.mkdir(exist_ok=True)
# Optional authoring audit helper;the exported source also checks its own two equal panels.
if len(sys.argv)>1:
 sys.path.insert(0,sys.argv[1])
 from audit_panel_alignment import require_matplotlib_panel_alignment
else:
 def require_matplotlib_panel_alignment(fig,json_out,overlay_svg,tolerance_pt=1.5,strict=True):
  fig.canvas.draw();rects=[a.get_position() for a in fig.axes if a.axison];height=fig.get_figheight()*72;width=fig.get_figwidth()*72
  diffs=[abs(rects[0].y0-r.y0)*height for r in rects[1:]]+[abs(rects[0].y1-r.y1)*height for r in rects[1:]]+[abs(rects[0].width-r.width)*width for r in rects[1:]]
  ok=max(diffs,default=0)<=tolerance_pt
  Path(json_out).write_text(json.dumps(dict(verdict='PASS' if ok else 'FAIL',auditable=True,summary=dict(fail=0 if ok else 1),maximum_difference_pt=max(diffs,default=0),scope='equal row plot-area edges and widths',tolerance_pt=tolerance_pt)))
  if strict and not ok:raise AssertionError('rendered panel edges or widths differ')
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams.update({'font.family':['Songti SC','Arial'],'font.size':9.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'legend.frameon':False,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9})
INK='#253536';TEAL='#376f66';BLUE='#577798';GOLD='#a2763b';PALE='#eff4f2';GRAY='#718180'
def save(fig,name,diagram=False):
 if diagram:(OUT/(name+'.alignment.json')).write_text(json.dumps(dict(status='NOT APPLICABLE',reason='single diagram canvas;no comparable quantitative axes')))
 else:require_matplotlib_panel_alignment(fig,json_out=str(OUT/(name+'.alignment.json')),overlay_svg=str(OUT/(name+'.alignment.svg')),tolerance_pt=1.5,strict=True)
 fig.savefig(OUT/(name+'.pdf'))
 fig.savefig(OUT/(name+'.svg'))
 fig.savefig(OUT/(name+'.png'),dpi=300)
 plt.close(fig)
def canvas(h):
 f,a=plt.subplots(figsize=(6.3,h));f.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,label,size=9.5,fill=PALE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.009',lw=.7,edgecolor='#aebeb8',facecolor=fill));a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK,linespacing=1.4)
def arrow(a,p,q,dashed=False):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,color=TEAL if not dashed else GRAY,lw=.8,linestyle='--' if dashed else '-',connectionstyle='arc3'))
f,a=canvas(3.65)
box(a,.015,.79,.28,.17,'数据准备\n官方 ZIP → 校验与同文分组')
box(a,.365,.79,.27,.17,'开发训练\n训练13497 / 开发1503')
box(a,.71,.79,.275,.17,'官方dev评价资料\n完整5000 / 其共同子集500')
arrow(a,(.305,.875),(.355,.875))
box(a,.365,.53,.27,.16,'模型与配置资产\nD0词典 / N0、N1权重')
arrow(a,(.5,.78),(.5,.70))
box(a,.015,.26,.28,.17,'输入与服务\n原文校验 / 本机任务调度')
box(a,.365,.26,.27,.17,'抽取与定位\n本地D0/N0/N1 / API L0/L3')
box(a,.71,.26,.275,.17,'评估与证据\n严格指标 / 分层 / 失败记录')
arrow(a,(.305,.345),(.355,.345));arrow(a,(.5,.52),(.5,.44));arrow(a,(.645,.345),(.70,.345))
arrow(a,(.848,.78),(.848,.44),True);a.text(.868,.62,'参考标注\n仅供评分',fontsize=8.7,ha='left',va='center',color=GRAY)
box(a,.19,.015,.62,.135,'统一实体或错误状态 → 实体可视化 / JSON导出',size=10)
arrow(a,(.5,.25),(.5,.16))
a.text(.02,.07,'本机 CPU',fontsize=8.5,color=GRAY);a.text(.83,.07,'外部 API独立边界',fontsize=8.5,color=GRAY)
save(f,'architecture',True)
f,a=canvas(4.2)
box(a,.25,.875,.50,.095,'输入：id + 原始文本（不带参考实体）',size=10)
box(a,.25,.705,.50,.10,'校验文本与路线 → 保留 Unicode 原文索引')
arrow(a,(.5,.865),(.5,.815))
box(a,.015,.47,.45,.15,'本地分支\nD0：词典匹配 / 分词边界\nN0/N1：字符窗口 → 跨度打分',size=9)
box(a,.535,.47,.45,.15,'API分支\n原文 + 固定零/三样本提示\n响应 → JSON与出现序号校验',size=9)
arrow(a,(.35,.695),(.24,.63));arrow(a,(.65,.695),(.76,.63))
box(a,.015,.27,.45,.115,'D0：保留匹配原文跨度\nN0/N1：阈值解码 → 窗口回映\n右开终点 / 按位置与类别去重',size=9)
box(a,.535,.27,.45,.115,'精确子串 + 出现序号 → 字符跨度\n非法内容或传输失败 → 明确错误状态',size=9)
arrow(a,(.24,.46),(.24,.395));arrow(a,(.76,.46),(.76,.395))
box(a,.20,.08,.60,.11,'输出：九类实体、原文、[start,end) 或错误\n实体表 / 嵌套可视化 / JSON导出',size=9.5)
arrow(a,(.24,.26),(.36,.20));arrow(a,(.76,.26),(.64,.20))
a.text(.50,.018,'离线评分另读取参考标注；预测分支不接收金标',ha='center',fontsize=9,color=GRAY)
save(f,'data-flow',True)
d=json.loads((ROOT/'runs/stage6-supplement-20261020/cpu-threshold/summary.json').read_text());results=d['results']
with (OUT/'threshold-source.csv').open('w') as out:
 w=csv.writer(out);w.writerow(['method','threshold','tp','predicted','gold','precision','recall','f1','character_f1'])
 for r in results:
  m=r['metrics']['micro'];w.writerow([r['method'],r['threshold'],m['tp'],m['predicted'],m['gold'],m['precision'],m['recall'],m['f1'],r['metrics']['character_type_micro']['f1']])
f,axes=plt.subplots(1,2,figsize=(6.3,2.65));f.subplots_adjust(left=.095,right=.98,bottom=.23,top=.86,wspace=.36)
for method,color in [('N0',BLUE),('N1',TEAL)]:
 rs=[r for r in results if r['method']==method];levels=[r['threshold'] for r in rs];ms=[r['metrics']['micro'] for r in rs]
 axes[0].plot(levels,[100*m['f1'] for m in ms],color=color,marker='o',ms=3.5,lw=1.4,label=method)
 axes[1].plot([100*m['recall'] for m in ms],[100*m['precision'] for m in ms],color=color,marker='o',ms=3.5,lw=1.2,label=method)
 zero=next(r['metrics']['micro'] for r in rs if r['threshold']==0)
 axes[1].plot([100*zero['recall']],[100*zero['precision']],marker='o',ms=6,markerfacecolor='white',markeredgecolor=color,lw=0)
axes[0].axvline(0,color=GRAY,ls='--',lw=.7,zorder=0);axes[0].set(xlabel='解码阈值',ylabel='严格实体F1 / %',ylim=(55,74),xticks=[-1,-.5,0,.5,1],title='a  阈值与开发质量')
axes[1].set(xlabel='严格实体召回率 / %',ylabel='严格实体精确率 / %',xlim=(43,83),ylim=(47,85),title='b  精确率与召回率取舍')
axes[0].legend(fontsize=9,loc='lower left')
for a in axes:a.grid(axis='y',color='#e5eae7',lw=.5);a.set_axisbelow(True)
save(f,'threshold-tradeoff')
print('Exported 3 draft figures:architecture,data-flow,threshold-tradeoff;PDF/SVG/300dpi PNG')
