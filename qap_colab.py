import requests
lines=str(requests.get('http://anjos.mgi.polymtl.ca/qaplib/data.d/chr12a.dat').content)[2:]
lines=lines.split('\n')[0].split('\\n')

import numpy as np
m=int((lines[0]).split()[0])
n=m
A=lines[2:(2+m)]
A=np.array([i.split() for i in A])
B=lines[(3+m):(3+2*m)]
B=np.array([np.array(i.split()).astype('float64') for i in B])
A=np.array([int(i) for i in A.reshape([m*n])]).reshape([m,n])
B=np.array([int(i) for i in B.reshape([m*n])]).reshape([m,n])
#A=A/(A.mean())
#B= B/(B.mean())
structure=np.array([7,5,12,2,1,3,9,11,10,6,8,4]) #([26,15,11,7,4,12,13,2,6,18,1,5,9,21,8,14,3,20,19,25,17,10,16,24,23,22])-1
#TT=np.kron(A,B)
TT=np.kron(A,B)
#TT=TT+1*abs(min(np.linalg.eigvals(TT)))*np.diag(np.ones(m*n))
#TT=(TT-TT.min())/(TT.max()-TT.min())
L=np.zeros(m*n)
for i in range(m):
    L[(i*n)+structure[i]]=1
L1=L.reshape([m,n])


from scipy.stats import binom 
import numpy as np
import copy
N=500
p=np.array([[1/n]*n for _ in range(m)])
def S(x):
  return -x.T.dot(TT).dot(x)
rho=20
eps=2e-6
jitter=0.5e-1
x=[]
score=[]
for _ in range(N):
  tmp0=[np.random.choice(list(range(n)),size=1,p=p[i]/sum(p[i]))[0] for i in range(m)]
  tmp=np.zeros([m,n])
  for i in range(m):
    tmp[i][tmp0[i]]=1
  x.append(tmp)
  score.append(S(tmp.reshape(m*n)))
x=np.array(x)
elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
p=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)/len(elite)+jitter*np.random.random([m,n])
t=1
while t<200:#sum([min(1-p[i],p[i]) for i in range(n)])>eps:
  x=[]
  score=[]
  for _ in range(N):
    tmp0=[np.random.choice(list(range(n)),size=1,p=p[i]/sum(p[i]))[0] for i in range(m)]
    tmp=np.zeros([m,n])
    for i in range(m):
      tmp[i][tmp0[i]]=1
    x.append(tmp)
    score.append(S(tmp.reshape(m*n)))
  x=np.array(x)
  elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
  tmp=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)
  elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
  p=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)/len(elite)+jitter*np.random.random([m,n])
  t+=1
  if t%20==0:
    print(t,sum((np.array([np.argmax(i) for i in p.reshape([m,n])])==structure))/m)

    
    
    
    
    
    
    
    
    
  import requests
lines=str(requests.get('http://anjos.mgi.polymtl.ca/qaplib/data.d/chr12a.dat').content)[2:]
lines=lines.split('\n')[0].split('\\n')

import numpy as np
m=int((lines[0]).split()[0])
n=m
A=lines[2:(2+m)]
A=np.array([i.split() for i in A])
B=lines[(3+m):(3+2*m)]
B=np.array([np.array(i.split()).astype('float64') for i in B])
A=np.array([int(i) for i in A.reshape([m*n])]).reshape([m,n])
B=np.array([int(i) for i in B.reshape([m*n])]).reshape([m,n])
#A=A/(A.mean())
#B= B/(B.mean())
structure=np.array([7,5,12,2,1,3,9,11,10,6,8,4])-1 #([26,15,11,7,4,12,13,2,6,18,1,5,9,21,8,14,3,20,19,25,17,10,16,24,23,22])-1

TT=np.kron(A,B)
L=np.zeros(m*n)
for i in range(m):
    L[(i*n)+structure[i]]=1
L1=L.reshape([m,n])
from scipy.stats import binom 
import numpy as np
import copy
N=500
p=np.array([[1/n]*n for _ in range(m)])
def S(x,y):
  if len(np.unique(y))>=m/1.5:
    return -x.T.dot(TT).dot(x)
  else:
    return -x.T.dot(TT).dot(x)-1000000
rho=10
eps=2e-6
jitter=0.5e-1
x=[]
score=[]
for _ in range(N):
  tmp0=[np.random.choice(list(range(n)),size=1,p=p[i]/sum(p[i]))[0] for i in range(m)]
  tmp=np.zeros([m,n])
  for i in range(m):
    tmp[i][tmp0[i]]=1
  x.append(tmp)
  score.append(S(tmp.reshape(m*n),tmp0))
x=np.array(x)
elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
#eliteset={}
#for i in elite:
 # eliteset[S(x[i[0]].reshape(m*n))]=x[i[0]]
p=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)/len(elite)+jitter*np.random.random([m,n])#np.sum(np.array(list(eliteset.values())),axis=0)/len(elite)+jitter*np.random.random([m,n])
t=1

while t<200:#sum([min(1-p[i],p[i]) for i in range(n)])>eps:
  x=[]
  score=[]
  for _ in range(N):
    tmp0=[np.random.choice(list(range(n)),size=1,p=p[i]/sum(p[i]))[0] for i in range(m)]
    tmp=np.zeros([m,n])
    for i in range(m):
      tmp[i][tmp0[i]]=1
    x.append(tmp)
    score.append(S(tmp.reshape(m*n),tmp0))
  x=np.array(x)
  elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
  tmp=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)
  elite=np.argwhere(np.array(score)>=np.percentile(sorted(score),100-rho))
  #for i in elite:
   # eliteset[S(x[i[0]].reshape(m*n))]=x[i[0]]
  p=np.sum(x[elite].reshape([len(elite),m,n]),axis=0)/len(elite)+jitter*np.random.random([m,n])  #np.sum(np.array(list(eliteset.values())),axis=0)/len(elite)+jitter*np.random.random([m,n])
  t+=1
  if t%2==0:
    print(t,sum((np.array([np.argmax(i) for i in p.reshape([m,n])])==structure))/m)
