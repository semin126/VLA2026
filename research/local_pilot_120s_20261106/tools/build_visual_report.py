"""Build a portable, source-backed student report from the completed 30-trial pilot.

Only recorded images/video/trace clocks are used. No new policy rollouts or model fitting.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT=Path('C:/Windows/Fonts/malgun.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams['axes.unicode_minus']=False
PALETTE={'A':'#647fa1','B':'#62967b','C':'#b47b42'}
LABEL={'A':'A','B':'B','C':'C1'}
CASES=[('oneway','scene_0030','B와 C1이 같은 물체를 정리한 사례'),
       ('mutual','scene_0050','C1에서만 정확도가 증가한 사례'),
       ('mutual','scene_0090','B에서만 정확도가 증가한 반대 사례'),
       ('oneway','scene_0040','세 조건 모두 정확도가 증가하지 않은 사례')]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def cp(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(source)==sha(target),'Refusing to replace different source evidence: '+str(target)
    else:shutil.copyfile(source,target)
def font(size):return ImageFont.truetype(str(FONT),size) if FONT.exists() else ImageFont.load_default()
def text(draw,xy,value,size=22,fill='#263442'):draw.text(xy,str(value),font=font(size),fill=fill)
def nums(names):return ', '.join(n.removeprefix('object_')+'번' for n in names) or '없음'
def signed(value):return f'{value*100:+.0f}%p'

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--skip-comparisons',action='store_true')
args=ap.parse_args();SRC=args.source.resolve();OUT=args.out.resolve()
assert SRC!=OUT
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'.gitattributes').write_text('# Preserve exact recorded bytes so media_manifest SHA256 also matches Git downloads.\ntrials/** -text whitespace=cr-at-eol\ndata/** -text whitespace=cr-at-eol\nassets/** -text\nvideos/** -text\nreport.pdf -text\n',encoding='utf-8')
def original(name):
    return next(p for p in (SRC/name,SRC/'data'/name,SRC/'assets'/name) if p.is_file())
summary=json.loads(original('report_summary.json').read_text(encoding='utf-8'))
assert summary['finished'] and summary['reported_trials']==30
audit=json.loads(original('audit.json').read_text(encoding='utf-8'))
assert audit['reports']==30
rows=[];by={}
for path in sorted((SRC/'trials').glob('*/scene_*/42/*/report.json')):
    family,scene,seed,condition=path.parts[-5:-1]
    r=json.loads(path.read_text(encoding='utf-8'))
    assert r['status']=='completed' and not r['diagnostic']
    dst=OUT/'trials'/family/scene/seed/condition
    for item in path.parent.iterdir():
        if item.is_file() and (item.name in ['report.json','before.png','after.png','process.mp4','scene.json',
                'events.jsonl','task.json','strategy_used.py','starter_strategy_used.txt'] or item.name.startswith('plan_')):
            cp(item,dst/item.name)
    spec=json.loads((path.parent/'scene.json').read_text(encoding='utf-8'))
    if (path.parent/'trace.npz').is_file():
        with np.load(path.parent/'trace.npz') as trace:
            indices=list(range(0,len(trace['wall']),4))
            if indices[-1]!=len(trace['wall'])-1:indices.append(len(trace['wall'])-1)
            clock={'steps':indices,'wall_seconds':[float(trace['wall'][i]) for i in indices],
                'fps':5,'source':'Original render_video stride4, including final recorded state; no interpolation.'}
    else:clock=json.loads((path.parent/'frame_clock.json').read_text(encoding='utf-8'))
    write_json(dst/'frame_clock.json',clock)
    gain=[n for n,v in r['final']['objects'].items() if v['correct'] and not r['initial']['objects'][n]['correct']]
    loss=[n for n,v in r['initial']['objects'].items() if v['correct'] and not r['final']['objects'][n]['correct']]
    row={'family':family,'scene':scene,'condition':condition,'label':LABEL[condition],
        'k':len(spec['research_targets']),'initial':r['initial']['acc'],'acc':r['final']['acc'],
        'gain':r['final']['acc']-r['initial']['acc'],'f1':r['final']['macro_f1'],
        'seconds':r['summary']['total_wall_seconds'],'calls':r['summary']['planner_calls'],
        'steps':r['summary']['steps'],'scene_success':r['final']['scene_success'],'gained':gain,'lost':loss,
        'base':dst.relative_to(OUT).as_posix(),'r':r,'clock':clock}
    rows.append(row);by[family,scene,condition]=row
assert len(rows)==30
for name in ['report_summary.json','audit.json','initial_image_comparison.json','plan.json','progress.json',
             'report_table.csv','trial_metrics.csv','REPORT_METHOD.md']:
    cp(original(name),OUT/'data'/name)
(OUT/'assets').mkdir(exist_ok=True)
cp(original('accuracy_f1.png'),OUT/'assets/accuracy_f1.png')
cp(original('accuracy_f1.pdf'),OUT/'assets/accuracy_f1.pdf')
print('Copied all30 unchanged source videos and before/after images.',flush=True)

SCENES=[(f,s) for f in ['oneway','mutual'] for s in sorted({r['scene'] for r in rows if r['family']==f})]
def image(path):return Image.open(path).convert('RGB')
def montage(family,scene):
    src=by[family,scene,'A'];w=360;gap=16;top=94;bottom=98
    canvas=Image.new('RGB',(4*w+5*gap,top+w+bottom),'#f4f6f8');d=ImageDraw.Draw(canvas)
    text(d,(gap,14),f'{family} / {scene} — 정리 대상 {src["k"]}개 (전체 큐브 10개)',26)
    panels=[('공통 시작 상태','A','before.png')]+[(LABEL[c]+' 종료 상태',c,'after.png') for c in 'ABC']
    for i,(title,c,name) in enumerate(panels):
        x=gap+i*(w+gap);row=by[family,scene,c]
        text(d,(x,57),title,22)
        canvas.paste(image(OUT/row['base']/name).resize((w,w),Image.Resampling.LANCZOS),(x,top))
        acc=row['initial'] if i==0 else row['acc']
        text(d,(x,top+w+8),f'ACC {acc:.0%}'+('' if i==0 else ' / 변화 '+signed(row['gain'])),21)
        text(d,(x,top+w+41),'시작부터 정리된 큐브도 포함' if i==0 else '새로 정리: '+nums(row['gained']),17)
        if i>0:text(d,(x,top+w+67),f'VLM {row["calls"]}회 / {row["seconds"]:.1f}초',17)
    path=OUT/'assets'/f'{family}_{scene}_before_after.png';canvas.save(path)
    return path.relative_to(OUT).as_posix()
montages={pair:montage(*pair) for pair in SCENES}

# Whole-test overview, and per-scene gains that expose the high initial ACC.
overview=Image.new('RGB',(5*254+32,2*310+70),'white');d=ImageDraw.Draw(overview)
text(d,(16,10),'평가에 사용한 10개 고정 씬 — 결과와 무관하게 미리 선택',25)
for n,(family,scene) in enumerate(SCENES):
    r=by[family,scene,'A'];x=16+(n%5)*254;y=56+(n//5)*310
    text(d,(x,y),f'{family} / 대상 {r["k"]}개',17)
    overview.paste(image(OUT/r['base']/'before.png').resize((238,238)),(x,y+29))
    text(d,(x,y+270),f'{scene} / 시작 ACC {r["initial"]:.0%}',15)
overview.save(OUT/'assets/scene_overview.png')
matrix=np.array([[by[f,s,c]['gain']*100 for c in 'ABC'] for f,s in SCENES])
fig,ax=plt.subplots(figsize=(8,6.1))
ax.imshow(matrix,vmin=-10,vmax=10,cmap='RdYlGn',aspect='auto')
ax.set_xticks([0,1,2],['A','B','C1'])
ax.set_yticks(range(10),[f'{f} {s} (대상 {by[f,s,"A"]["k"]}개)' for f,s in SCENES],fontsize=9)
for y in range(10):
    for x in range(3):ax.text(x,y,f'{matrix[y,x]:+.0f}%p',ha='center',va='center',fontsize=11)
ax.set_title('씬별 ACC 증가량 - 최종 ACC - 시작 ACC\n전체 30회, 씬을 제외하지 않은 결과',fontsize=13,pad=12)
fig.tight_layout();fig.savefig(OUT/'assets/acc_gain_by_scene.png',dpi=200);fig.savefig(OUT/'assets/acc_gain_by_scene.pdf');plt.close(fig)

def planner_spans(row):
    spans=[]
    for e in row['r']['events']:
        if e['event']=='plan_received':
            plan=json.loads((OUT/row['base']/f'plan_{e["call"]:03d}.json').read_text(encoding='utf-8'))
            end=e['wall'];spans.append((max(0,end-plan['seconds']),end,e['call']))
    return spans
def timeline(family,scene):
    fig,ax=plt.subplots(figsize=(10,2.6))
    for i,c in enumerate('ABC'):
        r=by[family,scene,c];end=r['seconds']
        ax.broken_barh([(0,min(end,120))],(i-.28,.56),facecolors='#7d9fb5')
        for start,stop,call in planner_spans(r):
            ax.broken_barh([(start,max(0,min(stop,120)-start))],(i-.28,.56),facecolors='#de9d51')
            ax.text((start+min(stop,120))/2,i,str(call),ha='center',va='center',fontsize=8)
        if end<120:ax.broken_barh([(end,120-end)],(i-.28,.56),facecolors='#e6e9ec')
        ax.text(121,i,f'ACC {r["initial"]:.0%}→{r["acc"]:.0%} / {r["steps"]}스텝',va='center',fontsize=9)
    ax.set_yticks([0,1,2],['A','B','C1']);ax.invert_yaxis();ax.set_xlim(0,163)
    ax.set_xticks(range(0,121,20));ax.set_xlabel('실제 과제 경과 시간(초)');ax.set_title(f'{family} / {scene}: 주황=VLM 추론, 파랑=나머지 실행, 회색=종료 후',fontsize=11)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    name=f'assets/{family}_{scene}_timeline.png';fig.savefig(OUT/name,dpi=180);plt.close(fig)
    return name
timelines={(f,s):timeline(f,s) for f,s,_ in CASES}

def source_frames(row):
    reader=imageio.get_reader(OUT/row['base']/'process.mp4')
    frames=[frame for frame in reader];reader.close()
    assert len(frames)==len(row['clock']['wall_seconds']),(row['base'],len(frames),len(row['clock']['wall_seconds']))
    return frames
comparisons={};comparison_meta=[]
for family,scene,title in CASES:
    name=f'videos/{family}_{scene}_wall_comparison.mp4';path=OUT/name;path.parent.mkdir(exist_ok=True)
    comparisons[family,scene]=name
    comparison_meta.append({'file':name,'family':family,'scene':scene,'fps':5,'frames':601,
        'clock_range_seconds':[0,120],'video_seconds':120.2,
        'mapping':'Nearest recorded preceding frame by trace.wall; inference/ended intervals hold the recorded image. Final sample included. No motion interpolation or synthetic frames.'})
    if args.skip_comparisons:continue
    if path.exists():
        print('Preserving completed comparison clip',name,flush=True);continue
    print('Rendering wall-clock comparison',family,scene,flush=True)
    rr=[by[family,scene,c] for c in 'ABC'];frames=[source_frames(r) for r in rr]
    spans=[planner_spans(r) for r in rr]
    tmp=path.with_name(path.stem+'.part.mp4')
    with imageio.get_writer(tmp,fps=5,codec='libx264',quality=7,macro_block_size=16,
            ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart']) as writer:
        for tick in range(601):
            wall=tick/5;canvas=Image.new('RGB',(1440,384),'#f4f6f8');d=ImageDraw.Draw(canvas)
            for j,r in enumerate(rr):
                times=np.asarray(r['clock']['wall_seconds'])
                idx=int(np.clip(np.searchsorted(times,wall,side='right')-1,0,len(times)-1))
                canvas.paste(Image.fromarray(frames[j][idx]).resize((464,267)),(j*480+8,44))
                text(d,(j*480+10,8),f'{r["label"]} / 공통 시계 {wall:05.1f}s',23)
                status='종료 후 최종 프레임 유지' if wall>=r['seconds'] else ('VLM 추론 대기' if any(a<=wall<b for a,b,_ in spans[j]) else '로봇 실행 / VLA 추론 포함')
                text(d,(j*480+10,319),status,18)
                text(d,(j*480+10,347),f'최종 ACC {r["acc"]:.0%} / 총 VLM {r["calls"]}회',17)
            writer.append_data(np.asarray(canvas))
    tmp.replace(path)
    del frames
write_json(OUT/'data/comparison_video_method.json',{'scope':'Derived playback only; original process.mp4 files remain byte-identical.',
    'cases':comparison_meta,'note':'The short original videos follow simulation time and omit VLM pauses. Comparison clips use trace wall clocks, hold frames during pauses and after a trial ends; 601 samples at5fps =120.2sec including endpoint.'})

def metrics_table():
    head=['계열','조건','시작 ACC','최종 ACC','증가','macro F1','씬 완료','평균 과제 시간','VLM 호출']
    vals=[]
    for g in summary['groups']:
        assert g['completed_n']==5 and not g['error_n']
        vals.append([g['family'],g['condition'],f'{g["initial_acc_mean"]:.0%}',f'{g["acc_mean"]:.0%}',
            signed(g['acc_gain_mean']),f'{g["macro_f1_mean"]:.3f}','0/5',f'{g["total_wall_seconds_mean"]:.1f}초',f'{g["planner_calls_mean"]:.1f}회'])
    return head,vals
def md_table(head,vals):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(v)+' |' for v in vals])
def html_table(head,vals):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in v)+'</tr>' for v in vals)+'</tbody></table></div>'
CASE_NOTES={
 ('oneway','scene_0030'):'시작 ACC는20%였다. A는20%를 유지했고, B와 C1은8번 큐브를 새로 정리해30%가 됐다. C1은 VLM을2회 호출했지만 B(1회)보다 최종 ACC가 높지는 않았다. 세 조건 모두 전체 정리는 완료하지 못했다.',
 ('mutual','scene_0050'):'시작 ACC는80%였다. A와 B는80%를 유지했고 C1은8번 큐브를 새로 정리해90%가 됐다. 이 씬에서 C1이 B보다 나았다는 관찰은 맞지만, 이를 전체 전략의 우수성으로 확대하지 않는다.',
 ('mutual','scene_0090'):'처음에는10개 모두 정리가 필요했다. B는4번 큐브를 새로 정리해10%가 됐지만 A와 C1은0%였다. C1만 개선된 앞 사례와 반대되는 결과로, 평균이 같아도 개선된 씬은 다를 수 있음을 보여 준다. 추가 추론 시간과 계획 내용이 함께 달라 원인을 하나로 단정할 수 없다.',
 ('oneway','scene_0040'):'처음에는10개 모두 정리가 필요했고 A·B·C1 모두 최종 ACC0%였다. 코드 오류가 아니라 제한 시간 안에 정확도를 높이지 못한 결과다. 본문에 개선 사례만 남기지 않기 위해 이 실패 사례를 함께 제시한다.'}
METHOD='''두 계열(oneway·mutual)에서 미리 고정한 씬을5개씩 골랐다. 각 계열은 정리가 필요한 큐브가2·4·6·8·10개인 씬을 하나씩 포함하며, 전체 큐브 수는 항상10개다. 같은 씬·시드42에 A·B·C1을 적용해30회를 평가했다. GPU는 로컬 RTX3090Ti, 시간 제한은120초, 운반 단위는240스텝, 관측 간격은60스텝, 최대 VLM 호출은12회다. VLM R5와 부분 작업 VLA motion7k를 고정하고 세 조건 모두 같은 VLA를 썼다. A는 전체 과제 지시를 받으며 VLM을 호출하지 않는다. B는 VLM 초기 계획1회를 실행하고, C1은 학생의 실패 의심 전략에 따라 재계획한다.'''
LIMITS=[
 '학생 Colab300초 실험과 별도인 로컬120초 예비 실험이다. 모델·장비·시간 조건이 다른 결과를 한 평균으로 합치지 않는다.',
 '계열별 독립 씬은5개, 정책 시드는1개다. 30회 실행을30개 독립 씬으로 쓰지 않는다. 같은 번호 큐브를50번 관측했다고 독립 표본50개로 세지 않는다.',
 'A에도 부분 작업으로 학습한 VLA를 사용했다. 전체 과제 지시의 학습 분포 차이가 있어, 별도로 전체 과제를 학습한 A와의 공정한 우열 비교는 아니다.',
 'oneway는90도 회전 큐브를 포함하고 mutual은 서로 막는 쌍을 포함한다. 막힘 방식과 회전의 영향을 이 두 묶음만으로 분리할 수 없다.',
 '정답 좌표를 쓰는 D로 씬을 선정했다. D의 선정 점수를 학습 모델과의 독립적인 경쟁 성능으로 해석하지 않는다.',
 'B/C1의 초기 물리 상태·프롬프트·첫 VLM 응답은10쌍 모두 같다. 실제 VLM 영상은9쌍이 완전히 같고 mutual scene_0050의 RGB채널2개만 각각1/255 차이가 있었다. 원본과 차이 검사를 보존했다.',
 '새로 정리된 큐브 번호는 실행 후 정답 물리 상태로 채점했다. 이 채점 결과를 VLM/VLA의 실행 입력으로 제공한 것은 아니다.'
]
TAKEAWAY='''30회 모두 코드 오류 없이 종료했지만 전체 씬 정리 완료는 모든 계열·조건에서0/5회였다. 시작 ACC 평균40%에 대해 A는40%, B·C1은42%였으므로 증가량은 각각0·2·2%p다. 이번 예비 실험에서 C1의 추가 ACC 이득은 관찰되지 않았고 평균 macro F1도 B보다 낮았다. C1은 B보다 VLM을 더 자주 호출했고 평균 과제 시간도 약15.7초(oneway),17.7초(mutual) 길었다. 이는 이번 모델·씬·예산에서의 관찰이며 피드백 루프가 일반적으로 무의미하다는 증거는 아니다.'''
VIDEO_NOTE='''원본 process.mp4는 시뮬레이션 시간 기준이며 VLM 추론 대기를 그대로 재생하지 않는다. 영상 자막의 wall은 실제 과제 시간이다. 별도 A/B/C1 비교 영상은 trace.wall로0~120초를 맞춰 VLM 대기 중에는 직전 기록 프레임, 종료 뒤에는 최종 프레임을 유지한다. 실제로 없던 동작을 보간하지 않았다. 끝점까지 포함해 비교 영상 길이는 약120.2초다. 점수는 원본 실험의120초 제한 결과를 그대로 쓴다.'''
head,vals=metrics_table()
md=['# 학생용 시각 보고서 — 로컬120초 A·B·C1 예비 실험','',
    '**2026-11-06 / oneway5씬 + mutual5씬 / 총30회 / 실행 오류0건**','',
    '> 처음 읽을 때: 시작부터 정리된 큐브가 있다. 최종 ACC42%를 새로42% 정리했다는 뜻으로 쓰면 안 된다. 시작40%에서2%p 증가한 결과다.','',
    '[브라우저 보고서](index.html) · [인쇄용 PDF](report.pdf) · [요약 CSV](data/report_table.csv) · [실행별 CSV](data/trial_metrics.csv)','',
    '## 1. 무엇을 비교했나','',METHOD,'',
    '| 조건 | 계획과 실행 |','|---|---|','| A | 전체 지시 → 공통 VLA. VLM 호출0회 |','| B | 초기 VLM 계획1회 → 공통 VLA |','| C1 | B와 같은 첫 계획 → 실행 후 실패 의심 시 재계획 |','',
    '![미리 고정한10개 씬](assets/scene_overview.png)','',
    '## 2. 전체 결과','',md_table(head,vals),'',
    '값은 각 씬에서 계산한 점수의 산술평균이다. ACC는 정답 구역에 안정적으로 놓인 물체 비율, macro F1은 두 목표 구역의 F1 평균이다. 시간은 추론·제어를 포함하고 모델 로딩·영상 저장은 제외한다. 짧게 끝난 미완료 실행도 포함하므로 성공까지의 시간으로 해석하지 않는다.','',
    '![전체30회의 평균과개별값](assets/accuracy_f1.png)','',
    '![모든씬의ACC증가량](assets/acc_gain_by_scene.png)','',
    TAKEAWAY,'',
    '## 3. 실행 전후와 영상으로 확인하기','',
    '본문 사례는 관찰된 네 패턴(둘 다 개선·C1만 개선·B만 개선·모두 미개선)을 설명하려고 고른 것이다. 평균은 사례만이 아니라30회 전부로 계산했다. 전체10씬 그림과30회 영상은 아래 부록에 모두 제공한다.','',VIDEO_NOTE,'']
case_html=[]
for n,(family,scene,title) in enumerate(CASES,1):
    note=CASE_NOTES[family,scene]
    md += [f'### 사례{n}. {title} — {family}/{scene}','',note,'',f'![실행전후비교]({montages[family,scene]})','',
        f'![실제경과시간타임라인]({timelines[family,scene]})','',f'[A/B/C1 실제시간 동기화 비교 영상]({comparisons[family,scene]})','']
    case_html.append(f'<section class="case" id="case{n}"><h3>사례{n}. {html.escape(title)}</h3><p class="eyebrow">{family} / {scene}</p><p>{note}</p><figure><img src="{montages[family,scene]}" alt="공통 시작 상태와 A/B/C1 종료 상태"><figcaption>녹화된 실제512px 전후 그림. 새로 정리된 번호는 최종 물리 상태 채점값이다.</figcaption></figure><figure><img src="{timelines[family,scene]}" alt="과제 시간별 VLM 추론과 나머지 실행"><figcaption>주황 구간은 응답 시각에서 기록된 추론 시간을 빼 복원했다. 기록 시점의 수ms 오차가 있을 수 있다. 파랑에는 VLA 추론·제어·관측이 포함된다.</figcaption></figure><div class="screen-only"><video controls preload="none" poster="{montages[family,scene]}" src="{comparisons[family,scene]}"></video><p><a href="{comparisons[family,scene]}">동기화 비교 영상 파일</a> · 세 조건의 공통 시계0~120초, 대기·종료 구간은 프레임 유지</p></div></section>')
md+=['## 4. 보고서에 적을 수 있는 문장','',
    '> 본 예비 실험에서는 두 계열 모두 B와 C1의 최종 ACC 평균이42%로 같았으며, 시작 ACC40%에서 각각2%p 증가했다. C1은 일부 씬에서 B보다 높은 ACC를 보였지만 반대 사례도 존재했고, 전체 평균 ACC의 추가 개선은 관찰되지 않았다. C1의 추가 VLM 호출 비용과 하위 행동 모델의 제한을 고려해, 이번 소규모 결과만으로 피드백 루프의 일반적인 효과를 단정하지 않았다.','',
    '## 5. 해석의 한계','']+['- '+v for v in LIMITS]+['','## 부록. 전체10씬과30회 원본 영상','',
    '모든 실행의 점수·전후 그림·원본 영상·report.json을 제공한다. 과정이 좋지 않다는 이유로 제외한 씬은 없다.','']
for family,scene in SCENES:
    md+=[f'### {family} / {scene} — 정리 대상{by[family,scene,"A"]["k"]}개','',f'![전체조건전후]({montages[family,scene]})','']
    vv=[]
    for c in 'ABC':
        r=by[family,scene,c];b=r['base']
        vv.append([r['label'],f'{r["initial"]:.0%}→{r["acc"]:.0%}',signed(r['gain']),f'{r["f1"]:.3f}',
            f'[원본 영상]({b}/process.mp4)',f'[전]({b}/before.png) / [후]({b}/after.png)',f'[기록]({b}/report.json)'])
    md += [md_table(['조건','ACC','증가','F1','영상','그림','원본'],vv),'']
md+=['## 자료 출처와 사용법','',
    '숫자는 data/report_summary.json 및 각 trials/.../report.json, 전후 그림은 before.png/after.png, 영상은 process.mp4에서 가져왔다. 모든 원본 복사 파일의 SHA256은 data/media_manifest.json에 기록했다. 생성 그림은 원본을 배치하거나 원본 수치를 시각화한 것이며 모델을 다시 실행하거나 점수를 바꾸지 않았다.','',
    '영상과 필터를 함께 보려면 폴더 전체를 내려받아 index.html을 브라우저에서 연다. README나PDF만 복사하면 연결된 영상이 함께 가지 않는다. PNG/PDF 그림은 assets/에서 보고서에 삽입하고, 인용 시 로컬120초 예비 실험·GPU·씬 수를 적는다.','',
    '그림·HTML·Markdown 생성기: tools/build_visual_report.py (조교의 Windows Python3.12 환경에서 실행). PDF는 브라우저 인쇄로 만들었으며, 배포 폴더를 입력으로 재생성하면 기존 PDF도 복사한다. 원본 실험 소스 커밋은 '+audit['code_commit']+'이다.','',
    '배포 폴더에서 재생성하는 예:','',
    '```powershell','python tools/build_visual_report.py --source . --out ../local_pilot_rebuilt','```']
(OUT/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

public_rows=[{k:v for k,v in r.items() if k not in ['r','clock']} for r in rows]
write_json(OUT/'data/gallery.json',public_rows)
cards=[]
for r in public_rows:
    b=r['base']
    if (OUT/b/'events.jsonl').is_file():event_link=f'<a href="{b}/events.jsonl">사건 로그</a>'
    else:
        assert not by[r['family'],r['scene'],r['condition']]['r']['events']
        event_link='기록된 사건 없음(실행 오류 아님)'
    cards.append(f'<article class="trial-card" data-family="{r["family"]}" data-condition="{r["label"]}"><h4>{r["family"]} / {r["scene"]} / {r["label"]}</h4><p>정리 대상{r["k"]}개 · ACC {r["initial"]:.0%}→{r["acc"]:.0%} ({signed(r["gain"])}) · F1 {r["f1"]:.3f}</p><video controls preload="none" poster="{b}/before.png" src="{b}/process.mp4"></video><p>{r["seconds"]:.1f}초 / VLM {r["calls"]}회 / {r["steps"]}스텝 · 새로 정리: {nums(r["gained"])}</p><div class="pair"><figure><img loading="lazy" src="{b}/before.png" alt="실행 전"><figcaption>전</figcaption></figure><figure><img loading="lazy" src="{b}/after.png" alt="실행 후"><figcaption>후</figcaption></figure></div><p><a href="{b}/process.mp4">원본 MP4</a> · <a href="{b}/report.json">채점 기록</a> · {event_link}</p></article>')
full_head=['계열','씬','조건','시작→종료ACC','증가','F1','과제시간','VLM호출','새로 정리']
full_vals=[[r['family'],r['scene'],r['label'],f'{r["initial"]:.0%}→{r["acc"]:.0%}',signed(r['gain']),f'{r["f1"]:.3f}',f'{r["seconds"]:.1f}초',str(r['calls']),nums(r['gained'])] for r in rows]
css='''
:root{--ink:#223340;--muted:#586975;--blue:#315e78;--line:#dce4e8;--paper:#fff}*{box-sizing:border-box}body{margin:0;background:#edf1f3;color:var(--ink);font:16px/1.75 'Malgun Gothic',sans-serif}main{max-width:1200px;margin:28px auto;background:white;padding:42px 54px;border-radius:16px}h1{font-size:34px;line-height:1.35;letter-spacing:-1px}h2{margin-top:48px;border-bottom:2px solid var(--line);padding-bottom:8px;font-size:25px}h3{font-size:21px;margin-bottom:4px}h4{margin:0;font-size:17px}a{color:var(--blue)}.eyebrow{color:var(--muted);font-size:13px;margin:0}.lead{font-size:18px}.notice{background:#edf5f7;border-left:4px solid #5b8b9b;padding:16px 20px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric{padding:15px;background:#f4f6f7;border-radius:9px}.metric strong{display:block;font-size:26px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6;margin:16px 0}th,td{border-bottom:1px solid var(--line);padding:9px 8px;text-align:left;white-space:nowrap}th{background:#edf2f4}figure{margin:20px 0}img{max-width:100%;height:auto}figcaption{font-size:12px;color:var(--muted)}video{width:100%;max-height:440px;background:#17212a;border-radius:8px}.case{margin:30px 0;padding-top:10px}.toolbar{position:sticky;top:0;background:white;border:1px solid var(--line);padding:12px;display:flex;gap:14px;align-items:center;z-index:2;flex-wrap:wrap}.toolbar select,.toolbar button{font:inherit;padding:5px 10px;border:1px solid #b7c7d0;background:#fff;border-radius:5px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.trial-card{border:1px solid var(--line);border-radius:10px;padding:14px}.trial-card p{font-size:13px;margin:8px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pair figure{margin:7px 0}.pair img{width:100%}.hidden{display:none!important}small{color:var(--muted)}.footer{margin-top:40px;font-size:12px;color:var(--muted)}@media(max-width:760px){main{margin:0;padding:22px 16px;border-radius:0}h1{font-size:27px}.metrics{grid-template-columns:1fr 1fr}.gallery{grid-template-columns:1fr}}@page{size:A4;margin:16mm 14mm}@media print{body{background:white;font-size:10pt;line-height:1.6}main{max-width:none;margin:0;padding:0;border-radius:0}h1{font-size:23pt}h2{font-size:17pt;margin-top:20pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}table{font-size:8pt}th,td{padding:5px 4px;white-space:normal}figure{break-inside:avoid;margin:12pt 0}p,li{orphans:3;widows:3}.screen-only,.gallery,.toolbar{display:none!important}.case{break-before:page}.metrics{break-inside:avoid}.table-wrap{overflow:visible}video{display:none}a{color:#315e78}.footer{font-size:8pt}.raw-table{break-before:page}}
'''
page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>학생용 시각 보고서 | 로컬120초 A·B·C1</title><style>'''+css+'''</style></head><body><main>
<p class="eyebrow">2026-11-06 · 고정 모델 R5 + motion7k · 로컬 RTX3090Ti</p><h1>같은 장면에서 A·B·C1은<br>얼마나 정리했을까?</h1><p class="lead">oneway·mutual 각5씬으로 수행한 로컬120초 예비 실험의 수치·전후 그림·실행 영상을 모았습니다.</p>
<div class="metrics"><div class="metric"><strong>10씬 × 3조건</strong>독립 씬10개, 총30회</div><div class="metric"><strong>120초</strong>실행당 과제 제한</div><div class="metric"><strong>0건</strong>실험 실행 오류</div><div class="metric"><strong>0 / 30회</strong>전체 씬 정리 완료</div></div>
<p class="notice"><b>시작 ACC 평균은40%였습니다.</b> 최종42%는 새로42%를 정리했다는 뜻이 아니라 <b>시작점보다2%p 증가</b>한 값입니다. 학생 Colab300초 결과와 별도 표로 보고하세요.</p>
<p class="screen-only"><a href="report.pdf">인쇄용PDF</a> · <a href="README.md">Markdown 보고서</a> · <a href="data/report_table.csv">요약CSV</a> · <a href="data/trial_metrics.csv">30회CSV</a> · <a href="#gallery">전체 영상30개</a></p>
<h2>1. 실험 방법</h2><p>'''+METHOD+'''</p>'''+html_table(['조건','동작'],[['A','전체 과제 지시 → VLA / VLM0회'],['B','초기 VLM 계획1회 → VLA'],['C1','B와 같은 첫 계획 → 실패 의심 시 재계획']])+'''<figure><img src="assets/scene_overview.png" alt="고정10씬"><figcaption>결과를 보기 전에 고정한10개 씬. 전체 큐브는항상10개이고 정리 대상 수만 달라집니다.</figcaption></figure>
<h2>2. 전체 결과</h2>'''+html_table(head,vals)+'''<p>각 씬 점수의 산술평균입니다. macro F1은 두 목표 구역의 F1 평균입니다. 시간은 추론·제어를 포함하고 모델 로딩·영상 저장은 제외합니다. 계획 소진으로 일찍 끝난 미완료 실행도 포함하므로 성공까지의 시간과 다릅니다.</p><figure><img src="assets/accuracy_f1.png" alt="평균ACC F1과각씬점수"><figcaption>막대는5씬 평균, 점은각씬 점수, 점선은시작ACC평균입니다.</figcaption></figure><figure><img src="assets/acc_gain_by_scene.png" alt="전체30회ACC증가량"><figcaption>시작 ACC를 빼야 실제로 정확도가 증가한 씬을 알 수 있습니다. 전체30회를 빠짐없이 표시했습니다.</figcaption></figure><p class="notice">'''+TAKEAWAY+'''</p>
<h2>3. 전후 그림과 영상으로 확인하기</h2><p>아래 사례는 네 결과 패턴을 설명하려고 선정했습니다. 통계는 사례만이 아니라 전체30회로 계산했습니다. 개선된 사례와 반대·미개선 사례를 함께 보세요.</p><p>'''+VIDEO_NOTE+'''</p>'''+''.join(case_html)+'''
<h2>4. 보고서에 적을 수 있는 문장</h2><blockquote>본 예비 실험에서는 두 계열 모두 B와 C1의 최종 ACC 평균이42%로 같았으며 시작 ACC40%에서 각각2%p 증가했다. C1은 일부 씬에서 B보다 높은 ACC를 보였지만 반대 사례도 존재했고, 전체 평균 ACC의 추가 개선은 관찰되지 않았다. 추가 VLM 호출 비용과 하위 행동 모델의 제한을 고려해, 소규모 결과만으로 피드백 루프의 일반적인 효과를 단정하지 않았다.</blockquote>
<h2>5. 해석의 한계</h2><ul>'''+''.join('<li>'+html.escape(v)+'</li>' for v in LIMITS)+'''</ul>
<section class="raw-table"><h2>부록 A. 30회 실행별 수치</h2>'''+html_table(full_head,full_vals)+'''</section>
<section class="screen-only" id="gallery"><h2>부록 B. 전체30회 영상·전후 그림</h2><p>원본 영상은 시뮬레이션 시간 재생입니다. 실제 경과 시간은 영상의wall자막과report.json을 확인하세요.</p><div class="toolbar"><label>계열 <select id="family"><option value="all">모두</option><option>oneway</option><option>mutual</option></select></label><label>조건 <select id="condition"><option value="all">모두</option><option>A</option><option>B</option><option>C1</option></select></label><button id="pause">모든 영상 일시정지</button><span id="count">30 / 30회</span></div><div class="gallery">'''+''.join(cards)+'''</div></section>
<p class="footer">출처: 원본report.json·before/after.png·process.mp4·events.jsonl·trace.wall. 원본 복사 파일 SHA256: <a href="data/media_manifest.json">media_manifest.json</a>. 점수와 영상을 다시 골라 실행하지 않았습니다. 실행 기반 커밋 '''+audit['code_commit'][:12]+'''. PDF는 정적 인쇄본이며 영상은 폴더 전체의index.html에서 재생합니다.</p>
</main><script>const cards=[...document.querySelectorAll('.trial-card')];function filter(){let n=0;for(const c of cards){const show=(family.value==='all'||c.dataset.family===family.value)&&(condition.value==='all'||c.dataset.condition===condition.value);c.classList.toggle('hidden',!show);if(show)n++;else c.querySelector('video').pause()}document.getElementById('count').textContent=n+' / 30회'}const family=document.getElementById('family'),condition=document.getElementById('condition');family.addEventListener('change',filter);condition.addEventListener('change',filter);document.getElementById('pause').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>v.pause()));</script></body></html>'''
page=page.replace('</style>','@media print{.footer{display:none}}</style>')
page=page.replace('<section class="raw-table">','<p class="eyebrow">원본 실행 기반 커밋 '+audit['code_commit'][:12]+' · 원본 채점·그림·영상 및 SHA256은 함께 제공한 data/와 trials/에 보존했습니다.</p><section class="raw-table">')
(OUT/'index.html').write_text(page,encoding='utf-8')
if not (OUT/'report.pdf').exists() and (SRC/'report.pdf').exists():cp(SRC/'report.pdf',OUT/'report.pdf')
tools=OUT/'tools';tools.mkdir(exist_ok=True)
if Path(__file__).resolve()!=(tools/'build_visual_report.py').resolve():shutil.copyfile(__file__,tools/'build_visual_report.py')
evidence={}
for path in (OUT/'trials').rglob('*'):
    if path.is_file():evidence[path.relative_to(OUT).as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
write_json(OUT/'data/media_manifest.json',{'original_trial_files':evidence,
    'scope':'Trial images, videos, JSON and strategy copies are unchanged. frame_clock.json is derived from original trace.wall. Generated comparison/figures are in separate folders.'})
print(json.dumps({'status':'built','out':str(OUT),'trials':len(rows),'montages':len(montages),
    'original_mp4':30,'comparison_mp4':len(comparisons),'original_media_bytes':sum((OUT/r['base']/'process.mp4').stat().st_size for r in rows)},ensure_ascii=False),flush=True)
