import scipy.optimize as opt
import numpy as np
import copy
import time
#模拟数据
m=10
n=m
inputDim=[n,m]    #每层交换机个数,3:最高层个数，6：最底层个数；因为我们的推导是从顶层向底层推，所以我把array倒过来
D=[np.random.rand(inputDim[i+1]) for i in range(len(inputDim)-1)] #交换机个数为4、5、6的层的单个交换机延时

structure=[[np.random.randint(0, inputDim[i]-1) for _ in range(inputDim[i+1])] for i in range(len(inputDim)-1)]  
#structure=list(range(n))
#np.random.shuffle(structure)
#structure=[structure]
#structure: [[1, 2, 1, 1], [0, 3, 3, 3, 0], [1, 0, 4, 0, 3, 0]]  
#其中[1, 2, 1, 1]指的是交换机个数为4的层与交换机个数为3的层之间的交换机连接情况；其中的第i个数a[i]表示交换机个数为4的层中第i-1台交换机与换机个数为3的层中第a[i]台交换机相连
T=[np.zeros([inputDim[i],inputDim[i]]) for i in range(len(inputDim))]#各层的同级交换机间通信时间, 最开始展现的是PSW，最后print的是ASW
T[0]=np.random.rand(inputDim[0],inputDim[0])
for k in range(1,len(inputDim)):
    for i in range(inputDim[k]):
        for j in range(inputDim[k]):
            T[k][i][j]=D[k-1][i]+D[k-1][j]+T[k-1][structure[k-1][i]][structure[k-1][j]]
#临时矩阵
TTT=np.zeros([m,m,n,n])
for i in range(m):
    for j in range(m):
        d1=np.ones([n,n])*D[-1][i]
        d2=np.ones([n,n])*D[-1][j]
        t0=np.ones([n,n])*T[-1][i][j]
        TTT[i][j]=abs(np.array(T[0])+d1+d2-t0)

#T'矩阵
TT=np.zeros([m*n,m*n])
for i in range(m):
    for j in range(m):
        for i1,k in enumerate(range((i*n),((i+1)*n),1)):
            for i2,l in enumerate(range((j*n),((j+1)*n),1)):
                TT[k][l]=TTT[i][j][i1][i2]
TT=(TT-TT.min())/(TT.max()-TT.min())                
L=np.zeros(m*n)
for i in range(m):
    L[(i*n)+structure[-1][i]]=1
    
    
    
from scipy.stats import binom 
import numpy as np
import copy
N=100
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
while t<150:#sum([min(1-p[i],p[i]) for i in range(n)])>eps:
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
    print(t)
print(np.array([np.argmax(i) for i in p.reshape([m,n])])==structure)
