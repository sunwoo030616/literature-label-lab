import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.deps'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/literature-label-mpl')
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def read(n):return json.loads((ROOT/'results'/n).read_text())
def main():
 results=read('test_results.json');policies=read('locked_policies.json');audit=read('data_audit.json');boot=read('paired_bootstrap.json');errors=read('errors.json');bench=read('batch_benchmark.json');env=read('encoding.json');support=read('label_support.json')
 graph_results=read('graph_test_results.json');results+=graph_results
 names=['tfidf','biomedbert','relation','feature_graph'];display=['TF-IDF + LR','BiomedBERT + LR','+ Co-occurrence','+ Feature graph'];rows={n:[r for r in results if r['method']==n] for n in names};primary={n:next(r for r in rows[n] if r['seed']==17) for n in names}
 agg={n:{k:{'mean':float(np.mean([r[k] for r in rows[n]])),'std':float(np.std([r[k] for r in rows[n]],ddof=1))} for k in ['macro_f1','micro_f1','multi_topic_recall','hamming_loss']} for n in names}
 null=[r['macro_f1'] for r in results if r['method'].startswith('shuffled')];d=agg['relation']['macro_f1']['mean']-agg['biomedbert']['macro_f1']['mean'];better=d>0
 summary='라벨 관계 보정의 평균 Macro-F1 개선은 관측됐지만 안정성과 대조 조건을 함께 확인해야 한다.' if better else '라벨 관계 보정은 평균 Macro-F1을 개선하지 못했다. 복잡도를 더하는 것만으로 성능 향상은 보장되지 않았다.'
 if boot['ci95'][0]<=0<=boot['ci95'][1]:summary+=' seed17의 paired bootstrap 95% 구간은 0을 포함해 일관된 개선의 근거가 약하다.'
 gd=agg['feature_graph']['macro_f1']['mean']-agg['biomedbert']['macro_f1']['mean']
 summary+=f' 특징 이웃 그래프 방식의 평균 Macro-F1 차이는 {gd*100:+.2f}pp였다.'
 summary=f"TF-IDF의 평균 Macro-F1 {agg['tfidf']['macro_f1']['mean']:.4f}가 frozen BiomedBERT {agg['biomedbert']['macro_f1']['mean']:.4f}보다 높았다. "+summary
 save={'aggregate':agg,'relation_delta_macro_f1':d,'null_macro_mean':float(np.mean(null)),'null_macro_min':float(min(null)),'null_macro_max':float(max(null)),'interpretation':summary}
 (ROOT/'results/summary.json').write_text(json.dumps(save,ensure_ascii=False,indent=2))
 figdir=ROOT/'results/figures';figdir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
 fig,axs=plt.subplots(1,2,figsize=(10,3.2))
 for ax,key,title in zip(axs,['macro_f1','micro_f1'],['Macro-F1 (equal weight per topic)','Micro-F1 (pooled predictions)']):
  vals=[agg[n][key]['mean'] for n in names];sd=[agg[n][key]['std'] for n in names];ax.bar(display,vals,yerr=sd,capsize=4,color=['#9099aa','#286bdf','#f47c24','#39a69b']);ax.set_ylim(0,1);ax.set_title(title);ax.set_xticks(range(4),['TF-IDF\n+ LR','BiomedBERT\n+ LR','+ Label\nrelation','+ Feature\ngraph']);ax.tick_params(axis='x',labelsize=9)
  for i,v in enumerate(vals):ax.text(i,v+.035,f'{v:.3f}',ha='center')
 fig.suptitle('Held-out test | 3 training subsamples | bars: mean +/- sample SD',fontsize=11);fig.tight_layout();fig.savefig(figdir/'quality.png');plt.close(fig)
 labels=list(primary['relation']['per_label']);xx=np.arange(len(labels));fig,ax=plt.subplots(figsize=(10,3.5))
 for offset,n,col in [(-.18,'biomedbert','#286bdf'),(.18,'relation','#f47c24')]:ax.bar(xx+offset,[primary[n]['per_label'][l]['f1'] for l in labels],.36,label=n,color=col)
 ax.set_xticks(xx,['Treatment','Diagnosis','Prevention','Mechanism','Transmission','Forecasting','Case report'],rotation=15);ax.set_ylim(0,1);ax.legend();ax.set_title('Topic-level F1 | predesignated seed 17');fig.tight_layout();fig.savefig(figdir/'topics.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,3));bs=[1,4,8];ax.bar([str(b) for b in bs],[np.mean([r['documents_per_second'] for r in bench if r['batch']==b]) for b in bs],color='#286bdf');ax.set_xlabel('Batch size');ax.set_ylabel('Documents / second');ax.set_title('Encoder only | same 32 documents | 3 repeats');fig.tight_layout();fig.savefig(figdir/'batch.png');plt.close(fig)
 # risk-coverage from calibration-selected settings, without tuning on test
 fig,ax=plt.subplots(figsize=(6,3))
 for n in ['biomedbert','relation']:
  cfg=next(c for c in policies if c['seed']==17 and c['method']==n);curve=cfg['review_policy']['calibration_curve'];valid=[r for r in curve if r['sample_f1'] is not None]
  ax.plot([r['coverage'] for r in valid],[r['sample_f1'] for r in valid],label=n)
 ax.set_xlabel('Automated coverage (calibration split)');ax.set_ylabel('Retained sample-F1');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(figdir/'review.png');plt.close(fig)
 lines=['# 생물의학 문헌 분류: 라벨 관계의 효과와 실패 분석','', '실행일: 2026-11-04 (KST). 기존 개별연구의 이론 학습 이후 수행한 후속 실험.','', '## 결과부터',summary,'', '## 데이터와 평가 설계',f"공식 train {audit['train']['raw']:,}편 중 후보 5,000편에서 80%(4,000편)를 3회 비복원 추출했다. dev 1,000편은 tune 600 / calibration 400으로 분리했다. 공식 test 2,500편에서 초록 누락 11편·중복 1편을 제외한 {audit['test']['selected']:,}편을 평가했다. 테스트는 선택에 사용하지 않았다.",'','입력은 title+abstract만 사용했다. frozen BiomedBERT mean pooling(max 256 tokens)과 동일한 독립 Logistic Regression을 사용했다. 전면 미세조정이 아니다. TF-IDF도 학습 부분표본으로만 vocabulary를 학습했다. 스케일러·관계 행렬도 train으로만 계산했다.','', '## 주요 비교: 평균 ± 표본 표준편차','| 방법 | Macro-F1 | Micro-F1 | 다중주제 recall | Hamming loss |','|---|---:|---:|---:|---:|']
 for n in names:lines.append('| '+n+' | '+' | '.join(f"{agg[n][k]['mean']:.4f} ± {agg[n][k]['std']:.4f}" for k in ['macro_f1','micro_f1','multi_topic_recall','hamming_loss'])+' |')
 lines+=['','표준편차는 3개 학습 부분표본 사이의 변동이다. 독립된 3개 테스트셋이나 공식 benchmark 5-fold 점수가 아니다.','',f"정상 관계 vs BiomedBERT Macro-F1 평균 차이: {d:+.4f} ({d*100:+.2f} percentage points).",f"무작위 라벨 대응 5개 × 학습 부분표본 3개: Macro-F1 평균 {np.mean(null):.4f}, 범위 {min(null):.4f}~{max(null):.4f}. 각 무작위 조건도 동일한 tune 규칙을 적용했다. 이 15개를 독립적인 통계 표본으로 가정하지 않는다.",f"seed17 paired bootstrap(문헌 재표집 1,000회): 차이 95% 구간 [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]. 다중 검정 보정이나 독립 재현은 수행하지 않았다.",'','## 선택한 보정 강도','alpha=0이면 관계 보정을 채택하지 않은 것이다. 개발 목표와 달라도 그대로 남겼다.']
 for r in results:
  if r['method'] in ['relation','feature_graph','random_neighbors']:lines.append(f"- {r['method']}, seed {r['seed']}: alpha={r['alpha']}, Macro-F1={r['macro_f1']:.4f}")
 lines+=['','## 특징 이웃 그래프 추가 실험', 'train 임베딩의 cosine 이웃으로 라벨 정보를 가중 평균한 뒤 softmax 라벨 표현의 열 간 cosine 관계를 계산했다. 중심화 후 같은 확률 보정에 사용했다. k=5/15/30과 alpha는 tune에서만 선택했다. 원논문의 multi-instance/triplet 학습 재현은 아니다. 같은 구조에서 무작위 이웃을 사용하는 대조도 포함했다.', '검증 선택 조건이 늘어나 과적합 가능성이 있고, 그래프 방식의 추가 효과에 대한 독립 재현은 아직 없다.', *[f"- seed {r['seed']}, {r['method']}: k={r['k']}, alpha={r['alpha']}, Macro-F1={r['macro_f1']:.4f}, Micro-F1={r['micro_f1']:.4f}" for r in graph_results], '', '## 주제별 오류: 미리 지정한 seed17','| 주제 | test support | 기본 F1 | 관계 F1 | 관계 precision | 관계 recall |','|---|---:|---:|---:|---:|---:|']
 for l in labels:
  a=primary['biomedbert']['per_label'][l];b=primary['relation']['per_label'][l];lines.append(f"| {l} | {b['support']} | {a['f1']:.4f} | {b['f1']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} |")
 lines+=['','가능한 원인: 표본 수가 작은 라벨의 관계 추정 오차, 독립 분류기의 기존 오류 전파, 256-token 잘림, frozen 표현과 선형 분류기의 한계, train/test 주제 분포 차이. 이번 실험만으로 원인을 확정하지 않았다.','', '## 검토 보류: 추가 개발 기능','검증 calibration에서 문헌별 sample-F1 .85 이상·자동처리 coverage .30 이상을 목표로 임계값을 선택했다. test 성능으로 조정하지 않았다. score는 calibrated probability가 아니다.']
 for n in ['biomedbert','relation']:
  r=primary[n]['review'];lines.append(f"- {n}: calibration 목표 달성={r['target_met_on_calibration']}, test 자동처리={r['coverage']:.1%}, 보류 {r['review_count']}편, 자동처리 sample-F1={r['retained_sample_f1']:.4f}, 전체 sample-F1={r['all_sample_f1']:.4f}.")
 lines+=['','검증에서 목표를 충족했어도 테스트 retained sample-F1이 .85 미만이면 목표가 일반화되지 않은 것이다. 검토 보류의 효과와 별개로 품질 보장 실패로 기록한다. 자동처리 성능과 처리 범위는 교환관계다. 사람이 보류 문헌을 실제로 검토한 실험은 아니며 인간 검토 후 최종 정확성도 측정하지 않았다. 높은 margin도 정답을 보장하지 않는다.','', '## 실패 사례','아래 사례는 결과를 본 뒤 설명용으로 선정했다. 사전 고정된 예시나 인과적 설명이 아니다. 학습 정답 자체의 전문 검수도 수행하지 않았다.']
 changed=[e for e in errors if e['relation_changed_prediction']];chosen=(changed[:4]+[e for e in errors if not e['relation_changed_prediction']][:2])
 for e in chosen:lines+=['',f"### PMID {e['pmid']}",e['title'],f"- 정답: {', '.join(e['gold'])}",f"- 기본: {', '.join(e['baseline']) or '(없음)'}",f"- 관계 보정: {', '.join(e['relation']) or '(없음)'}",f"- 과잉 예측: {', '.join(e['false_positive']) or '(없음)'} / 누락: {', '.join(e['false_negative']) or '(없음)'}"]
 trans=read('error_transitions.json')
 lines+=['', '## 관계 보정이 고친 문헌과 망가뜨린 문헌', f"seed17 exact-match 기준: 오답→정답 {trans['exact_match_fixed_by_relation']}편, 정답→오답 {trans['exact_match_broken_by_relation']}편. 라벨이 달라진 문헌 {trans['documents_with_changed_labels']}편.", '유행 예측 주제는 TP 32편과 FN 9편이 그대로인데 FP가 20→29편으로 늘어 precision이 떨어졌다. 치료 주제는 FN 285→210편으로 줄었지만 FP는 114→173편으로 늘었다. 관계 보정은 누락과 과잉 예측의 교환관계를 만들었다.', '', '## 추론 배치 실험',f"실행 장치: {env['device']} / {env['platform']}. 동일한 32개 검증 문헌, batch 1/4/8, warm-up 후 각 3회. 인코더만 측정하며 네트워크 API 지연이나 동시 사용자 처리량이 아니다.",'| batch | 평균 초 | documents/s | RSS MB (마지막 측정) |','|---|---:|---:|---:|']
 for b in bs:
  rr=[r for r in bench if r['batch']==b];lines.append(f"| {b} | {np.mean([r['seconds'] for r in rr]):.3f} | {np.mean([r['documents_per_second'] for r in rr]):.2f} | {rr[-1]['rss_mb']:.0f} |")
 lines+=['','RSS는 CPU 프로세스의 관측 시점 메모리이며 peak GPU 메모리가 아니다. MPS allocator 메모리는 raw JSON에 별도로 기록했다. 배치 순서를 무작위화하지 않았고 background 시스템 상태를 완전히 통제하지 않았다.','', '## API 구현과 검증','제목/초록 입력 검증, 모델 1회 적재, 추론 동시 실행 제한(바쁠 때 429), 복수 주제·검토 권고·유사 훈련 문헌 반환. 유사 문헌은 예측 설명의 근거가 아니며 검색 품질의 정답 평가는 수행하지 않았다. API smoke 결과는 results/api_smoke.json을 확인한다.','', '## 선행 논문과의 경계','T-MDML은 bag 입력, 특징·라벨 공간의 graph, 전역/라벨별 metric 및 triplet 최적화를 사용한다. 이번 코드는 문헌 1벡터와 통계적 관계 후처리로 연구 질문을 축소했다. 논문 재현·성능 우위·새로운 metric learning을 주장하지 않는다. 새로 구현한 것은 대조·안정성·보류 평가 및 실행 API다.','', '## 재현 및 한계','실제 모델·버전·장치·가중치 해시는 results/encoding.json, 원본 해시는 data_audit.json, 문헌 ID는 split_manifest.json, 선택 정책은 locked_policies.json에 있다. 모델 및 데이터는 무료 공개 리소스를 다운로드하여 로컬에서 실행했고 유료 API를 사용하지 않았다.','학습 데이터는 공식 전체 중 제한된 부분표본이며, 테스트 중복·초록 누락을 제외했다. 단일 도메인·7라벨의 결과를 전체 과학문헌·금융문서·단백질 기능예측으로 일반화하지 않는다. calibration 표본이 작아 보류 정책 안정성은 추가 검증이 필요하다.','', '## 제출용 문장',f"개별연구에서 학습한 라벨 상관관계 개념을 생물의학 문헌 분류 실험으로 확장했습니다. 무료 BiomedBERT와 통계 분류기를 구현하고, {audit['test']['selected']:,}편의 별도 테스트에서 정상·무작위 관계 및 학습 표본 변동을 비교했습니다. 성능 향상 여부와 오류를 기록하고, 검토 보류 정책과 문헌 분류·탐색 API를 구현했습니다.",'', '이 문장은 실제 구현·실험 범위에 맞춘 초안이며, 본인이 코드·결과를 확인하고 설명할 수 있는 범위로 제출한다. 기존 이론 학습과 11월 후속 개발의 기간은 구분한다.','', '## 출처','- NCBI 공식 데이터: https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/','- 과제·평가 정의: https://github.com/ncbi/biocreative_litcovid','- 무료 모델: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract','- Kim, Kan, Yi. T-MDML. AAAI 2025, 17850–17857. 사용자 제공 원문.']
 (ROOT/'docs/REPORT.md').write_text('\n'.join(lines))
 # Three-page portfolio addendum, concise evidence with explicit limitations.
 out=ROOT/'output';out.mkdir(exist_ok=True);pdfmetrics.registerFont(TTFont('Korean','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
 pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='Korean',italic='Korean',boldItalic='Korean')
 W,H=842,595;c=canvas.Canvas(str(out/'portfolio.pdf'),pagesize=(W,H));c.setTitle('생물의학 문헌 분류 — 개별연구 후속 실험')
 orange='#F47C24';dark='#142236';muted='#5D6877'
 def text(x,y,s,size=12,color=dark):c.setFillColor(HexColor(color));c.setFont('Korean',size);c.drawString(x,y,s)
 def para(x,y,s,width=744,size=12,color=dark):
  p=Paragraph(s,ParagraphStyle('p',fontName='Korean',fontSize=size,leading=size*1.55,textColor=HexColor(color),wordWrap='CJK'));ww,hh=p.wrap(width,500);p.drawOn(c,x,y-hh);return y-hh
 def head(num,title,sub):
  c.setFillColor(HexColor(orange));c.rect(0,H-12,W,12,fill=1,stroke=0);text(42,550,f'RESEARCH EXTENSION  /  2026.11  /  {num}',10,orange);text(42,512,title,25);para(42,487,sub,size=11,color=muted)
 def foot(num):text(42,22,'LitCovid · frozen BiomedBERT · 무료 로컬 실행 · 이론 학습과 후속 구현 기간 구분',9,muted);text(775,22,str(num),9,muted)
 head('01','논문에서 배운 라벨 관계, 실제 문헌에서도 유효할까?','개별연구의 T-MDML·라벨 상관관계 학습을 잇는 생물의학 문헌 분류 및 검토 지원 실험')
 y=430
 y=para(42,y,'<b>문제</b>  논문 한 편에 여러 주제가 붙을 때, 독립적으로 분류하면 드문 주제를 놓칠 수 있다. 라벨 간 관계를 추가하는 것이 실제로 도움이 되는지 검증했다.')-18
 y=para(42,y,'<b>구현</b>  제목·초록 → 무료 BiomedBERT 임베딩 → 7개 주제 분류 → 관계 보정 → 검토 보류 및 유사 문헌 탐색 API')-20
 y=para(42,y,'<b>데이터</b>  학습 후보 5,000편에서 4,000편씩 3회 학습 / tune 600편 / calibration 400편 / 별도 테스트 2,488편. PMID·텍스트 해시 중복과 초록 누락을 정리했다.')-20
 y=para(42,y,'<b>비교 설계</b>  TF-IDF / BiomedBERT / BiomedBERT+정상 관계 / 무작위 관계 5종·특징 이웃 그래프·무작위 이웃. 분류기·평가 조건을 맞추고, 설정 선택을 완료한 뒤 테스트했다.')-20
 y=para(42,y,'<b>직접 확장한 부분</b>  무작위 대조·학습 표본 안정성·paired bootstrap·검토 보류 평가·배치 추론 측정·로컬 API를 구현했다.')-18
 para(42,y,'<b>범위</b>  T-MDML 전체 재현이 아니다. 원논문의 bag·triplet metric learning과 이번 문헌 벡터·관계 후처리를 명확하게 구분했다.',size=11,color=muted);foot(1);c.showPage()
 head('02','좋아진 결과와 나빠진 결과를 함께 기록','세 번의 학습 부분표본으로 비교한 held-out test 결과. 성능 향상을 전제하지 않았다.')
 c.drawImage(str(figdir/'quality.png'),42,223,width=750,height=240)
 summary_bottom=para(42,210,summary,size=11)
 b=primary['biomedbert'];r=primary['relation']
 para(42,min(135,summary_bottom-10),f"<b>미리 지정한 seed17</b>  Macro-F1: {b['macro_f1']:.4f} → {r['macro_f1']:.4f}<br/>차이의 paired bootstrap 95% 구간: [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] / 세 학습 부분표본의 평균 차이: {d*100:+.2f}pp",size=11)
 para(42,72,'공식 전체 학습 데이터·5-fold·SOTA 비교가 아니다. 데이터 선택 변동과 통계적 불확실성을 포함한 소규모 후속 실험이다.',size=10,color=muted);foot(2);c.showPage()
 head('03','실험을 검토 가능한 문헌 처리 백엔드로','자동 분류를 무조건 적용하지 않고, 사람에게 넘길 범위와 로컬 실행 비용을 함께 측정했다.')
 c.drawImage(str(figdir/'topics.png'),42,258,width=750,height=210)
 rp=primary['relation']['review'];fast=max(bs,key=lambda b:np.mean([r['documents_per_second'] for r in bench if r['batch']==b]));slow=np.mean([r['documents_per_second'] for r in bench if r['batch']==1]);fastv=np.mean([r['documents_per_second'] for r in bench if r['batch']==fast])
 y=237;y=para(42,y,f"<b>검토 보류</b>  calibration sample-F1 .85 목표 달성: {'예' if rp['target_met_on_calibration'] else '아니오'}. test 자동처리 {rp['coverage']:.1%}, 검토 보류 {rp['review_count']}편. 자동처리 sample-F1 {rp['retained_sample_f1']:.4f} / 전체 {rp['all_sample_f1']:.4f}. 테스트에서는 .85 목표에 미달했다.",size=11)-12
 y=para(42,y,f"<b>추론 측정</b>  동일 문헌 32개·3회 반복. batch1 {slow:.2f}편/초, 관측상 최고 batch{fast} {fastv:.2f}편/초. 인코더만 측정했으며 API 전체 지연이나 분산 서빙 성능이 아니다.",size=11)-12
 y=para(42,y,'<b>API</b>  입력 검증·모델 적재·동시 실행 제한·복수 주제·검토 권고·유사 문헌 반환. 유사 문헌은 탐색용이며 예측을 입증하는 근거로 제시하지 않는다.',size=11)-12
 para(42,y,'<b>다음 검증</b>  문장별 bag 표현, 실제 T-MDML 구성요소 재현, 다른 도메인·더 큰 학습 표본, 검토자 실사용 평가. 이번 결과와 후속 계획을 구분한다.',size=10,color=muted);foot(3);c.save()
 print('Created REPORT.md and output/portfolio.pdf')
if __name__=='__main__':main()
