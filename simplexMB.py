# Hector FC 
# Algoritmo Simplex 

import numpy as np 
def simplexMB(A,b,c, indB): 
  n = len(c)
  m = len(b)  
  iter = 0   
  while True:  
    indN =  [ind for ind in range(n)  if ind not in indB]    
    matB  = A[:,indB] 
    xsol =  np.zeros((n,1))
    xsol[indB] =  np.linalg.solve(matB,b)
    custoR = np.zeros(n)        
    Y = np.linalg.solve(matB,A[:,indN])
    custoR[indN] = c[indN] - np.dot(c[indB],Y)        
    iS =  np.argmin(custoR)      
    if custoR[iS] < 0:
      y = np.linalg.solve(matB,A[:,iS])        
      delta = np.amax(y)                    
      if delta < 0:  
        print("O problema é ilimitado")
        break
      else:          
        indy = np.where(y>0)        
        it  = np.min(indy[0]) 
        temp1 =  xsol[indB[it],0] / y[it]      
        iR = min(indy[0])                                
        for i in indy[0]:
          temp2 = min(temp1, xsol[indB[i],0]/y[i])                                        
          if temp2 <= temp1: 
            iR = i 
          temp1 = temp2                      
        indB = set(indB).difference(set([ indB[iR]] ))        
        indB = list(indB.union(set([iS])))        
        iter = iter + 1        
    else:
      valOpt = np.dot(xsol.T,c)
      print("Valor otimo: ", valOpt[0])    
      break
    
    if iter > 5:    
      print("Numero de iterações permitido")
      break 
  return [indB, xsol, valOpt] 

##################################
 
def simplex2F(A,b,c):
  n = len(c)
  m = len(b)  
  idN =  np.identity(m)  
  matA = np.append(A,idN,axis=1)
  vc = np.append(np.zeros(n),np.ones(m))
  indB =[ i for i  in  range(m+1,n+m)] 
  sol = simplexMB(matA,b,vc,indB)  
  solLP = simplexMB(A,b,c,sol[0])  
  return solLP
    
##################################

A = np.array([[2, 1, 2], 
              [3, 3, 1]])

b = np.array([[4],[3]])
c = np.array([4,1,1])

sol = simplex2F(A,b,c)
print(30*"*")
print("Base: ", sol[0])
print("Sol basica: ", sol[1])
print("Valor otimo: ", sol[2][0])
print(30*"*")


