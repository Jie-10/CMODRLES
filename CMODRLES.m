classdef CMODRLES < ALGORITHM 
% <2026> <multi/many> <real/binary/permutation><constrained/none> 
 
methods 
function main(Algorithm, Problem) 

    %% Initialize MP, AP and DP 
    MP = Problem.Initialization();
    AP = MP;
    DP = MP;
    FitnessMP = CalFitness(MP.objs, MP.cons, 0);
    FitnessAP = CalFitness(AP.objs);
    FitnessDP = FitnessMP;
    [N,M] = size(MP.objs);

    %% Generate reference vectors
    [W,N] = UniformPoint(N,M);
    NumW = size(W,1);

    %% Calculate minimum angular threshold
    AngleW = acos(1-pdist2(W,W,'cosine'));
    AngleW(eye(NumW) == 1) = inf;
    MinAngle = mean(min(AngleW,[],2))/2;

    %% Main loop 
    while Algorithm.NotTerminated2(MP,[]) 
 
        %% Parent selection and offspring generation 
        Zmin = min([MP.objs; AP.objs; DP.objs], [], 1); 
 
        MatingIndexMP = TournamentSelection(2, length(MP), FitnessMP); 
        %MatingIndexAP = TournamentSelection(2, length(AP), FitnessAP); 
        MatingIndexDP = TournamentSelection(2, length(DP), FitnessDP); 
        if rand > 0.5
            O1 = Neighbor_Pairing_Strategy(Problem, MP(MatingIndexMP), MP, Zmin); 
            %O2 = Neighbor_Pairing_Strategy(Problem, AP(MatingIndexAP), AP, Zmin); 
            O3 = Neighbor_Pairing_Strategy(Problem, DP(MatingIndexDP), DP, Zmin); 
        else
            O1 = OperatorDE(Problem, MP, MP(randperm(N)), MP(randperm(N)));
            %O2 = OperatorDE(Problem, AP, AP(randperm(N)), AP(randperm(N)));
            O3 = OperatorDE(Problem, DP, DP(randperm(N)), DP(randperm(N)));
        end

        %% Environmental selection 
        % MP: SPEA2-CDP 
        [MP, FitnessMP] = EnviromentSelect1([MP, O1, O3], N); 
 
        % AP: SPEA2 
        %[AP, FitnessAP] = EnviromentSelect2([AP, O1, O2, O3], N, MinAngle,W); 
 
        % DP: W + epsilon-relaxed SPEA2-CDP
        [DP, FitnessDP] = EnviromentSelect3([DP, O1, O3], N, MinAngle,W); 

    end 
end 
end 
end 
