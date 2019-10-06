# Learning Rate Study

001 : 그냥 lr = 0.001  
002 : stepLR, lr = 0.001, stepwise = 10, gamma = 0.1  
003 : 그냥 lr = 0.01  
004 : exponentialLR, lr = 0.01, gamma = 0.9  
005 : cosine annealing LR,  tmax = 50  
006 : reduceLR, lr = 0.01  
007 : cyclicLR, lr = 0.01~0.1  
008 : cosine annealing, tmax = 10  

lambdaLR : 내맘대로 lambda함수 설정해서 줄이기 가능 - 안해봄..Lambda는 어떤 함수로 설정해야 하는 것인가..  
stepLR : 설정한 stepsize마다 gamma가 learning rate에 곱해짐  
MultistepLR : stepsize 간격을 맘대로 정할 수 있음 (여러 input을 통해)  
ExponentialLR : 지수로 줄임 lr * gamma ^ epoch  
CosineAnnealingLR : SGDR에서 제안하는 cosine annealing인데, restart는 구현되지 않음  
ReduceLROnPlateau : loss가 업데이트 되지 않으면 lr을 줄임 => scheduler.step()에 loss가 들어가야 함  
CyclicLR : 주기적으로 lr이 상승했다가 하강  
