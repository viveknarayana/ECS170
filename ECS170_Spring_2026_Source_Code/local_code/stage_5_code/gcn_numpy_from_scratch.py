"""Pure-NumPy GCN matching the PyTorch starter; sparse-feature optimized.
Layer op:  H_out = A_hat . (H_in . W) + b   (associative; identical to (A_hat H)W).
Keeping H_in sparse for the input layer makes high-dim bag-of-words features cheap.
"""
import os, time
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

CACHE = os.path.join(os.path.dirname(__file__), "cache")

def load_cached(name):
    z = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    shape = tuple(int(s) for s in z["shape"])
    A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape).astype(np.float32)
    X = sp.csr_matrix(z["feat"].astype(np.float32))      # keep features sparse
    return A, X, z["y"].astype(np.int64), list(z["classes"])

def make_split(labels, n_train_per, n_test_per, n_val=500, seed=0):
    rng = np.random.RandomState(seed); n_cls = labels.max() + 1
    train, test = [], []
    for c in range(n_cls):
        ids = np.where(labels == c)[0]; rng.shuffle(ids)
        train.extend(ids[:n_train_per]); test.extend(ids[n_train_per:n_train_per + n_test_per])
    train = np.array(train); test = np.array(test)
    used = set(train.tolist()) | set(test.tolist())
    remaining = np.array([i for i in range(len(labels)) if i not in used]); rng.shuffle(remaining)
    return train, remaining[:n_val], test

def mm(H, W):
    return H.dot(W) if sp.issparse(H) else H @ W

def xavier(fin, fout, rng):
    lim = np.sqrt(6.0 / (fin + fout))
    return rng.uniform(-lim, lim, size=(fin, fout)).astype(np.float32)

def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))

class GCN:
    def __init__(self, dims, dropout=0.5, seed=42):
        self.rng = np.random.RandomState(seed); self.dropout = dropout
        self.W = [xavier(dims[i], dims[i+1], self.rng) for i in range(len(dims)-1)]
        self.b = [np.zeros(dims[i+1], np.float32) for i in range(len(dims)-1)]
        self.nL = len(self.W)
        self.mW=[np.zeros_like(w) for w in self.W]; self.vW=[np.zeros_like(w) for w in self.W]
        self.mb=[np.zeros_like(b) for b in self.b]; self.vb=[np.zeros_like(b) for b in self.b]
        self.t=0
    def forward(self, A, X, train=False):
        self.c={"A":A,"H":[X],"pre":[],"Z":[],"mask":[]}; H=X
        for l in range(self.nL):
            pre = mm(H, self.W[l]); self.c["pre"].append(pre)
            Z = A.dot(pre) + self.b[l]; self.c["Z"].append(Z)
            if l < self.nL-1:
                Hr = np.maximum(Z,0)
                if train and self.dropout>0:
                    keep=1-self.dropout
                    m=(self.rng.rand(*Hr.shape)<keep).astype(np.float32)/keep
                    Hr=Hr*m; self.c["mask"].append(m)
                else: self.c["mask"].append(None)
                H=Hr
            else: H=Z
            self.c["H"].append(H)
        self.logits=H; return H
    def loss_grad(self, idx, y, wd):
        A=self.c["A"]; logp=log_softmax(self.logits); n=len(idx)
        nll=-logp[idx,y[idx]].mean()
        P=np.exp(logp); dZ=np.zeros_like(self.logits)
        dZ[idx]=P[idx]; dZ[idx,y[idx]]-=1.0; dZ[idx]/=n
        gW=[None]*self.nL; gb=[None]*self.nL
        for l in reversed(range(self.nL)):
            gb[l]=dZ.sum(0)
            dpre=A.dot(dZ)
            H=self.c["H"][l]
            gW[l]=(H.T.dot(dpre) if sp.issparse(H) else H.T@dpre) + wd*self.W[l]
            if l>0:
                dH=dpre@self.W[l].T
                m=self.c["mask"][l-1]
                if m is not None: dH=dH*m
                dZ=dH*(self.c["Z"][l-1]>0)
        return nll,gW,gb
    def adam(self,gW,gb,lr,b1=0.9,b2=0.999,eps=1e-8):
        self.t+=1
        for l in range(self.nL):
            for g,m,v,p in ((gW[l],self.mW,self.vW,self.W),(gb[l],self.mb,self.vb,self.b)):
                m[l]=b1*m[l]+(1-b1)*g; v[l]=b2*v[l]+(1-b2)*g*g
                p[l]-=lr*(m[l]/(1-b1**self.t))/(np.sqrt(v[l]/(1-b2**self.t))+eps)
    def predict(self,A,X): return self.forward(A,X,False).argmax(1)

def evaluate(model,A,X,y,idx):
    pred=model.predict(A,X)[idx]; yt=y[idx]
    a=accuracy_score(yt,pred)
    pm,rm,fm,_=precision_recall_fscore_support(yt,pred,average="macro",zero_division=0)
    pw,rw,fw,_=precision_recall_fscore_support(yt,pred,average="weighted",zero_division=0)
    return {"acc":float(a),"prec_macro":float(pm),"rec_macro":float(rm),"f1_macro":float(fm),
            "prec_weighted":float(pw),"rec_weighted":float(rw),"f1_weighted":float(fw)}

PER = {"cora":(20,150),"citeseer":(20,200),"pubmed":(20,200)}

def run(name, hidden=(64,), dropout=0.5, lr=0.01, wd=5e-4, epochs=200, seed=42, split_seed=0, data=None, eval_every=1):
    A,X,y,classes = data if data is not None else load_cached(name)
    n_feat = X.shape[1]
    tr,va,te = make_split(y, *PER[name], n_val=500, seed=split_seed)
    dims=[n_feat, *hidden, len(classes)]
    model=GCN(dims, dropout=dropout, seed=seed)
    hist={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    vl=float("nan"); vaa=float("nan")
    t0=time.time()
    for ep in range(1,epochs+1):
        model.forward(A,X,True); nll,gW,gb=model.loss_grad(tr,y,wd)
        tra=(model.logits.argmax(1)[tr]==y[tr]).mean(); model.adam(gW,gb,lr)
        if ep % eval_every == 0 or ep == epochs:
            model.forward(A,X,False); vlp=log_softmax(model.logits)
            vl=float(-vlp[va,y[va]].mean()); vaa=float((model.logits.argmax(1)[va]==y[va]).mean())
        hist["train_loss"].append(float(nll)); hist["val_loss"].append(vl)
        hist["train_acc"].append(float(tra)); hist["val_acc"].append(vaa)
    el=time.time()-t0
    m=evaluate(model,A,X,y,te); m["final_train_loss"]=hist["train_loss"][-1]
    return {"dataset":name,"dims":dims,"hidden":list(hidden),"dropout":dropout,"lr":lr,
            "weight_decay":wd,"epochs":epochs,"n_nodes":int(X.shape[0]),
            "n_features":int(n_feat),"n_classes":len(classes),"classes":classes,
            "n_train":int(len(tr)),"n_test":int(len(te)),"elapsed":el,
            "metrics":m,"history":hist}
