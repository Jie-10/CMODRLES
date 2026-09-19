function Offspring = Neighbor_Pairing_Strategy(Problem,MatingPop,Pop,Zmin)

    Objs = MatingPop.objs;
    [Num,M] = size(Objs);
    Objs = (Objs - repmat(Zmin,Num,1));
    Objs = Objs./repmat(max(sqrt(sum(Objs.^2,2)),1e-12),1,M);
    
    Objs1 = Pop.objs;
    [Num2,M] = size(Objs1);
    Objs1 = (Objs1 - repmat(Zmin,Num2,1));
    Objs1 = Objs1./repmat(max(sqrt(sum(Objs1.^2,2)),1e-12),1,M);
    
    CosV = Objs * Objs1';
    %     CosV = CosV - 3*eye(Num,Num);
    
    [~,SInd] = sort(-CosV,2);
    
    Nr=min(10,length(Pop));
    %     Nr=min(Num,10);
    Neighbor = SInd(:,1:Nr);
    
    Mate1 = MatingPop;
    
    P = ones(Num,1);
    for i = 1:Num
        P(i) = Neighbor(i,randi(Nr));
    end
    
    Mate2=Pop(P);
    if rand > 0.5
        Offspring=OperatorGAhalf(Problem,[Mate1,Mate2]);
    else
        Offspring=OperatorGAhalf(Problem,[Mate1,Mate2],{1,20,1,1/Problem.D});
    end
end
