import numpy as np 

A = np.array([[2, 1, 2], 
            [3, 3, 1]])  #  

b = np.array([[4],[3]])
c = np.array([4, 1, 1])
indB = set(np.array([0,2]))    # base 

def simplexMB(A,b,c, indB):    # def  da função  simplex  
  n = len(c)                       
  m = len(b)                     
  iter = 0                     #   
  while True:  
    indN = set(range(n)) - indB 
    indLB = list(indB)
    indLN = list(indN)

    matB  = A[:,indLB] 
    xsol =  np.zeros((n,1))
    xsol[indLB] =  np.linalg.solve(matB,b)
    custoR = np.zeros(n)
    base_c = c[indLB]   
    Y = np.linalg.solve(matB,A[:,indLN])
    custoR[indLN] = c[indLN] - np.dot(c[indLB],Y)

    iS =  np.argmin(custoR)

    if custoR[iS] < 0:
      y = np.linalg.solve(matB,A[:,iS])        
      delta = np.amax(y)        
      if delta < 0:  
        print("O problema é ilimitado")
        break
      else:
        iR = np.where( xsol[indLB]/y>0 )           
        indB = (indB - set(iR[0]) ) | set([iS])      
      print("iteração: ",iter)
      print(xsol)
      valOpt = np.dot(xsol.T,c)    
      print("valor na base: ",valOpt[0])
      iter = iter + 1 
    else:
      print("iteração",iter)
      print("Solução encontrada")
      valOpt = np.dot(xsol.T,c)
      print(xsol)
      print("Valor otimo: ", valOpt[0])    
      break
              
    if iter > 5:    
      print("Numero de iterações permitido")
      break 
  return [indB,xsol,valOpt] 

solP = simplexMB(A,b,c,indB)
print(solP[0])
