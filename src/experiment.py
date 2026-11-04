"""Frozen biomedical encoder + leakage-controlled label relation experiments."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('MPLCONFIGDIR','/tmp/literature-label-mpl')
import csv, json, hashlib, time, platform, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,precision_recall_fscore_support,hamming_loss,accuracy_score,average_precision_score
import joblib
ROOT=Path(__file__).resolve().parents[1]
LABELS=['Treatment','Diagnosis','Prevention','Mechanism','Transmission','Epidemic Forecasting','Case Report']
def save(name,x):
 (ROOT/'results'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_data():
 seen_id=set();seen_text=set();splits={};audit={}
 for name,cap in [('train',5000),('dev',1000),('test',None)]:
  rows=list(csv.DictReader((ROOT/'data/raw'/f'{name}.csv').open()));kept=[];counts={'raw':len(rows),'missing_abstract':0,'duplicate':0}
  for r in rows:
   if not r['abstract'].strip():counts['missing_abstract']+=1;continue
   text=r['title'].strip()+' '+r['abstract'].strip();h=hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
   if r['pmid'] in seen_id or h in seen_text:counts['duplicate']+=1;continue
   seen_id.add(r['pmid']);seen_text.add(h)
   unknown=set(r['label'].split(';'))-set(LABELS)
   if unknown:raise ValueError(unknown)
   kept.append({'pmid':r['pmid'],'title':r['title'],'text':text,'labels':r['label'].split(';'),'sha256':h})
  if cap and len(kept)>cap:
   inds=np.random.default_rng(20261104).choice(len(kept),cap,replace=False);kept=[kept[i] for i in sorted(inds)]
  counts['selected']=len(kept);counts['source_sha256']=digest(ROOT/'data/raw'/f'{name}.csv');audit[name]=counts;splits[name]=kept
 save('data_audit.json',audit);save('split_manifest.json',{k:[{'pmid':r['pmid'],'sha256':r['sha256']} for r in v] for k,v in splits.items()})
 return splits

def yof(rows):return np.array([[int(l in r['labels']) for l in LABELS] for r in rows])
def get_encoder():
 torch.set_num_threads(4);torch.manual_seed(20261104)
 tok=AutoTokenizer.from_pretrained(ROOT/'models/biomedbert',local_files_only=True)
 model=AutoModel.from_pretrained(ROOT/'models/biomedbert',local_files_only=True).eval()
 device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device)
 return tok,model,device

def sync(device):
 if device=='mps':torch.mps.synchronize()

def encode(texts,tok,model,device,batch=16):
 chunks=[];start=time.perf_counter()
 with torch.inference_mode():
  for i in range(0,len(texts),batch):
   x=tok(texts[i:i+batch],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
   h=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
   pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1);chunks.append(pooled.cpu().numpy())
   if i%256==0:print('encoded',i,'/',len(texts),'seconds',round(time.perf_counter()-start,1),flush=True)
 return np.concatenate(chunks)

def relation(y):
 prior=y.mean(0);c=(y.T@y)/(y.sum(0)[:,None]+1e-9)-prior[None,:];np.fill_diagonal(c,0)
 return c,prior

def corrected(p,c,prior,alpha):return np.clip(p+alpha*(p-prior)@c,0,1)
def thresholds(y,p):
 grid=np.arange(.1,.81,.05)
 return np.array([max(grid,key=lambda t:f1_score(y[:,j],p[:,j]>=t,zero_division=0)) for j in range(y.shape[1])])
def fit_policy(y,p,c=None,prior=None):
 best=None
 for a in ([0,.25,.5,1,2] if c is not None else [0]):
  q=corrected(p,c,prior,a) if c is not None else p;t=thresholds(y,q);score=f1_score(y,q>=t,average='macro',zero_division=0)
  if best is None or score>best['tune_macro_f1']+1e-12:best={'alpha':a,'thresholds':t,'tune_macro_f1':score}
 return best

def metrics(y,p,t):
 pred=(p>=t).astype(int);pr,re,f,s=precision_recall_fscore_support(y,pred,zero_division=0);multi=y.sum(1)>1
 return {'micro_f1':f1_score(y,pred,average='micro',zero_division=0),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'hamming_loss':hamming_loss(y,pred),'subset_accuracy':accuracy_score(y,pred),'macro_ap':average_precision_score(y,p,average='macro'),'multi_topic_recall':float((y[multi]*pred[multi]).sum()/max(1,y[multi].sum())),'predicted_labels_mean':pred.sum(1).mean(),'per_label':{l:{'precision':pr[j],'recall':re[j],'f1':f[j],'support':int(s[j])} for j,l in enumerate(LABELS)}}

def row_f1(y,pred):
 return 2*(y*pred).sum(1)/np.maximum(1,y.sum(1)+pred.sum(1))
def confidence(p,t):return np.min(np.abs(p-t),axis=1)
def review_policy(y,p,t):
 conf=confidence(p,t);f=row_f1(y,p>=t);opts=[]
 for margin in np.linspace(0,.35,36):
  keep=conf>=margin
  opts.append({'margin':margin,'coverage':keep.mean(),'sample_f1':float(f[keep].mean()) if keep.any() else None})
 valid=[o for o in opts if o['coverage']>=.3 and o['sample_f1']>=.85]
 return {'target_met':bool(valid),'selected':max(valid,key=lambda o:o['coverage']) if valid else opts[0],'calibration_curve':opts}

def main():
 splits=load_data();ys={k:yof(v) for k,v in splits.items()};save('label_support.json',{k:dict(zip(LABELS,y.sum(0).tolist())) for k,y in ys.items()})
 allrows=sum(splits.values(),[]);cache=ROOT/'data/embeddings.npz'
 fingerprint=hashlib.sha256((''.join(r['sha256'] for r in allrows)+digest(ROOT/'models/biomedbert/config.json')+'mean256').encode()).hexdigest()
 tok,model,device=get_encoder();start=time.perf_counter()
 used_cache=cache.exists() and str(np.load(cache)['fingerprint'])==fingerprint
 if used_cache:emb=np.load(cache)['embeddings']
 else:
  emb=encode([r['text'] for r in allrows],tok,model,device);np.savez_compressed(cache,embeddings=emb,fingerprint=fingerprint)
 save('encoding.json',{'device':device,'elapsed_seconds':time.perf_counter()-start,'samples':len(allrows),'dimension':emb.shape[1],'max_length':256,'pooling':'attention-mask mean, frozen encoder','cache_used':bool(used_cache),'model_revision':'d673b8835373c6fa116d6d8006b33d48734e305d','model_weights_sha256':digest(ROOT/'models/biomedbert/pytorch_model.bin'),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'sklearn':__import__('sklearn').__version__,'python':platform.python_version(),'platform':platform.platform()})
 xs={};n=0
 for k,v in splits.items():xs[k]=emb[n:n+len(v)];n+=len(v)
 order=np.random.default_rng(20261104).permutation(len(ys['dev']));tune=order[:int(len(order)*.6)];cal=order[int(len(order)*.6):]
 # No test labels accessed by model fitting, relation estimation, threshold or policy selection.
 configs=[];artifacts={}
 for seed in [17,42,73]:
  inds=np.sort(np.random.default_rng(seed).choice(len(ys['train']),int(.8*len(ys['train'])),replace=False));y=ys['train'][inds]
  vec=TfidfVectorizer(max_features=20000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
  xt=vec.fit_transform([splits['train'][i]['text'] for i in inds]);xd=vec.transform([r['text'] for r in splits['dev']])
  tf=OneVsRestClassifier(LogisticRegression(C=1,max_iter=1000,solver='liblinear')).fit(xt,y)
  scaler=StandardScaler().fit(xs['train'][inds]);lr=OneVsRestClassifier(LogisticRegression(C=1,max_iter=1000,solver='liblinear')).fit(scaler.transform(xs['train'][inds]),y)
  p=lr.predict_proba(scaler.transform(xs['dev']));pt=tf.predict_proba(xd);c,prior=relation(y)
  methods={'tfidf':(pt,None),'biomedbert':(p,None),'relation':(p,c)}
  for shuf in range(5):
   perm=np.random.default_rng(100+shuf).permutation(len(LABELS));methods[f'shuffled_{shuf}']=(p,c[np.ix_(perm,perm)])
  for name,(probs,mat) in methods.items():
   policy=fit_policy(ys['dev'][tune],probs[tune],mat,prior);q=corrected(probs,mat,prior,policy['alpha']) if mat is not None else probs
   rp=review_policy(ys['dev'][cal],q[cal],policy['thresholds'])
   configs.append({'seed':seed,'method':name,**policy,'review_policy':rp,'calibration':metrics(ys['dev'][cal],q[cal],policy['thresholds'])})
  artifacts[seed]=(vec,tf,scaler,lr,c,prior)
  print('tuned seed',seed,flush=True)
 save('locked_policies.json',configs)
 # Policies are now frozen. Final test evaluation begins here.
 results=[];preds={}
 for cfg in configs:
  seed=cfg['seed'];name=cfg['method'];vec,tf,scaler,lr,c,prior=artifacts[seed]
  p=tf.predict_proba(vec.transform([r['text'] for r in splits['test']])) if name=='tfidf' else lr.predict_proba(scaler.transform(xs['test']))
  if name.startswith('shuffled'):
   perm=np.random.default_rng(100+int(name.split('_')[1])).permutation(len(LABELS));c=c[np.ix_(perm,perm)]
  if name=='relation' or name.startswith('shuffled'):p=corrected(p,c,prior,cfg['alpha'])
  t=cfg['thresholds'];m=metrics(ys['test'],p,t);conf=confidence(p,t);keep=conf>=cfg['review_policy']['selected']['margin']
  m['review']={'coverage':keep.mean(),'retained_sample_f1':float(row_f1(ys['test'],p>=t)[keep].mean()) if keep.any() else None,'all_sample_f1':float(row_f1(ys['test'],p>=t).mean()),'review_count':int((~keep).sum()),'target_met_on_calibration':cfg['review_policy']['target_met']}
  results.append({'seed':seed,'method':name,'alpha':cfg['alpha'],**m});preds[(seed,name)]=p>=t
  if seed==17:np.save(ROOT/'results'/f'probabilities_{name}.npy',p)
 save('test_results.json',results)
 # Paired bootstrap is uncertainty estimation, not model selection.
 rng=np.random.default_rng(20261104);delta=[];y=ys['test']
 for _ in range(1000):
  ix=rng.integers(0,len(y),len(y));delta.append(f1_score(y[ix],preds[(17,'relation')][ix],average='macro',zero_division=0)-f1_score(y[ix],preds[(17,'biomedbert')][ix],average='macro',zero_division=0))
 save('paired_bootstrap.json',{'comparison':'relation - biomedbert, seed17','iterations':1000,'mean':np.mean(delta),'ci95':np.quantile(delta,[.025,.975])})
 errors=[]
 for i,r in enumerate(splits['test']):
  b=preds[(17,'biomedbert')][i];c=preds[(17,'relation')][i]
  if not np.array_equal(c,y[i]):errors.append({'pmid':r['pmid'],'title':r['title'],'gold':r['labels'],'baseline':[l for j,l in enumerate(LABELS) if b[j]],'relation':[l for j,l in enumerate(LABELS) if c[j]],'relation_changed_prediction':bool(np.any(b!=c)),'false_positive':[l for j,l in enumerate(LABELS) if c[j] and not y[i,j]],'false_negative':[l for j,l in enumerate(LABELS) if y[i,j] and not c[j]]})
 save('errors.json',errors);vec,tf,scaler,lr,c,prior=artifacts[17]
 joblib.dump({'vectorizer':vec,'tfidf':tf,'scaler':scaler,'classifier':lr,'relation':c,'prior':prior,'policies':[a for a in configs if a['seed']==17],'labels':LABELS},ROOT/'models/classifiers.joblib')
 np.savez_compressed(ROOT/'data/search_index.npz',embeddings=xs['train']);save('search_documents.json',[{'pmid':r['pmid'],'title':r['title'],'labels':r['labels']} for r in splits['train']])
 # fixed warm-up, matched items for every batch; these documents are not used for model selection.
 import psutil
 texts=[r['text'] for r in splits['dev'][:32]];encode(texts[:2],tok,model,device,batch=2);bench=[]
 for batch in [1,4,8]:
  for rep in range(3):
   sync(device);t=time.perf_counter();out=encode(texts,tok,model,device,batch);sync(device);dt=time.perf_counter()-t
   bench.append({'batch':batch,'repeat':rep,'seconds':dt,'documents_per_second':len(texts)/dt,'rss_mb':psutil.Process().memory_info().rss/1024**2,'device_allocated_mb':torch.mps.current_allocated_memory()/1024**2 if device=='mps' else None})
 save('batch_benchmark.json',bench)
 print('COMPLETE',json.dumps([{'seed':r['seed'],'method':r['method'],'macro_f1':r['macro_f1'],'alpha':r['alpha']} for r in results if r['method'] in ['tfidf','biomedbert','relation']]),flush=True)
if __name__=='__main__':main()
