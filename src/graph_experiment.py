import json,time
import numpy as np
import joblib
from scipy.special import softmax
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from experiment import ROOT,LABELS,load_data,yof,fit_policy,corrected,metrics,review_policy,confidence,row_f1,save

def graph_matrix(x,y,k,random_graph=False):
 if random_graph:
  rng=np.random.default_rng(20261104);idx=np.empty((len(y),k),int)
  for i in range(len(y)):
   v=rng.choice(len(y)-1,k,replace=False);idx[i]=v+(v>=i)
  weights=np.ones((len(y),k))/k
 else:
  distances,indices=NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(x).kneighbors(x)
  idx=np.array([ii[ii!=i][:k] for i,ii in enumerate(indices)])
  similarities=np.einsum('nd,nkd->nk',x,x[idx]);weights=softmax(similarities,axis=1)
 s=softmax(y+np.einsum('nk,nkl->nl',weights,y[idx]),axis=1)
 c=normalize(s,axis=0).T@normalize(s,axis=0)
 np.fill_diagonal(c,0);c=c-c.sum(0)[None,:]/(len(LABELS)-1);np.fill_diagonal(c,0)
 return c

def main():
 splits=load_data();ys={k:yof(v) for k,v in splits.items()};emb=np.load(ROOT/'data/embeddings.npz')['embeddings'];xs={};start=0
 for k,v in splits.items():xs[k]=emb[start:start+len(v)];start+=len(v)
 order=np.random.default_rng(20261104).permutation(len(ys['dev']));tune=order[:600];cal=order[600:];configs=[];artifacts={}
 for seed in [17,42,73]:
  ids=np.sort(np.random.default_rng(seed).choice(len(ys['train']),int(.8*len(ys['train'])),replace=False));y=ys['train'][ids];scaler=StandardScaler().fit(xs['train'][ids]);lr=OneVsRestClassifier(LogisticRegression(C=1,max_iter=1000,solver='liblinear')).fit(scaler.transform(xs['train'][ids]),y)
  p=lr.predict_proba(scaler.transform(xs['dev']));prior=y.mean(0);x=normalize(xs['train'][ids])
  for random_graph in [False,True]:
   best=None
   for k in [5,15,30]:
    c=graph_matrix(x,y,k,random_graph);policy=fit_policy(ys['dev'][tune],p[tune],c,prior)
    if best is None or policy['tune_macro_f1']>best['tune_macro_f1']+1e-12:best={**policy,'k':k,'matrix':c}
   c=best.pop('matrix');q=corrected(p,c,prior,best['alpha']);rp=review_policy(ys['dev'][cal],q[cal],best['thresholds']);name='random_neighbors' if random_graph else 'feature_graph'
   configs.append({'seed':seed,'method':name,**best,'review_policy':rp});artifacts[(seed,name)]=(scaler,lr,c,prior)
   print('graph tuned',seed,name,best['k'],best['alpha'],flush=True)
 save('graph_locked_policies.json',configs);results=[]
 for cfg in configs:
  scaler,lr,c,prior=artifacts[(cfg['seed'],cfg['method'])];p=lr.predict_proba(scaler.transform(xs['test']));q=corrected(p,c,prior,cfg['alpha']);m=metrics(ys['test'],q,cfg['thresholds']);keep=confidence(q,cfg['thresholds'])>=cfg['review_policy']['selected']['margin']
  m['review']={'coverage':keep.mean(),'sample_f1':float(row_f1(ys['test'],q>=cfg['thresholds'])[keep].mean()) if keep.any() else None}
  results.append({'seed':cfg['seed'],'method':cfg['method'],'k':cfg['k'],'alpha':cfg['alpha'],**m})
 save('graph_test_results.json',results);print('GRAPH COMPLETE',[(r['seed'],r['method'],r['macro_f1']) for r in results],flush=True)
if __name__=='__main__':main()
