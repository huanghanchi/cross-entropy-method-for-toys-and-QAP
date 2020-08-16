from scipy.stats import binom 
import numpy as np
import copy
n=10
N=50
y=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#np.random.choice([0,1],size=n)
p=np.array([0.5]*n)
def S(x):
  return n-np.linalg.norm(x-y,ord=1)
rho=1
eps=1e-6
jitter=1e-1
x=[]
score=[]
for _ in range(N):
  tmp=[np.random.choice([0,1],size=1,p=[1-p[i],p[i]])[0] for i in range(n)]
  x.append(tmp)
  score.append(S(tmp))
x=np.array(x)
elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
p_old=p.copy()
tmp=x[elite].reshape([len(elite),n]).T
for i in range(n):
  p[i]=np.sum(tmp[i])/len(elite)
t=1

while t<30:#sum([min(1-p[i],p[i]) for i in range(n)])>eps:
  x=[]
  score=[]
  for _ in range(N):
    tmp=[np.random.choice([0,1],size=1,p=[1-p[i],p[i]])[0] for i in range(n)]
    x.append(tmp)
    score.append(S(tmp))
  x=np.array(x)
  elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
  tmp=x[elite].reshape([len(elite),n]).T
  for i in range(n):
    p[i]=max(min(np.sum(tmp[i])/len(elite),1-jitter),jitter)
  print(p)
  t+=1
  if t%20==0:
    print(t)
