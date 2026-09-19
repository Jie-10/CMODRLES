function Fitness = CalFitness(PopObj,PopCon,epsilon)
% Calculate the fitness of each solution

    N = size(PopObj,1);
    if nargin == 1
        CV = zeros(N,1);
    else
        CV = sum(max(0,PopCon),2);
        CV(CV < epsilon) = 0; %用于放松约束的时候计算CV值
    end
    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            if CV(i) < CV(j)
                Dominate(i,j) = true;
            elseif CV(i) > CV(j)
                Dominate(j,i) = true;
            else
                k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
                if k == 1
                    Dominate(i,j) = true;
                elseif k == -1
                    Dominate(j,i) = true;
                end
            end
        end
    end
    
    %% Calculate S(i)每个个体支配的个数
    S = sum(Dominate,2);
    
    %% Calculate R(i)支配第i个体的所有个体支配其他个体的个数
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% 计算拥挤距离
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    %每个个体的拥挤距离取第√(N+2)大的距离
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% 计算适应度
    Fitness = R + D';
end
