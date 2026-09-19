function [Population,Fitness] = EnviromentSelect3(Population,N,MinAngle,W)
% Diversity-oriented environmental selection

    %% Basic data
    Obj  = Population.objs;
    Con  = Population.cons;
    NumQ = length(Population);

    %% Calculate angles between solutions and reference vectors
    CosQW = 1-pdist2(Obj,W,'cosine');
    CosQW = min(max(CosQW,-1),1);

    AngleQ = acos(CosQW);
    AngleQ(isnan(AngleQ)) = pi/2;

    %% Determine solutions contained in each subspace
    InSubspaceQ = AngleQ <= MinAngle;

    %% Calculate fitness
    FitnessAll = CalFitness(Obj, Con, 0);

    %% Environmental selection
    selectedIndex   = zeros(1,N);
    selectedFitness = zeros(1,N);
    selected        = false(1,NumQ);
    selectedCount   = 0;

    %% Step 1: Select one solution from each subspace
    for i = 1:N

        if selectedCount >= N
            break;
        end

        %% Remaining solutions
        R = find(~selected);

        if isempty(R)
            break;
        end

        %% Solutions located in the current subspace
        Ti = find(InSubspaceQ(:,i)' & ~selected);

        if isempty(Ti)

            %% Empty subspace:
            % Select the nearest remaining solution
            [~,bestLocal] = min(AngleQ(R,i));
            x = R(bestLocal);

        else

            %% Non-empty subspace:
            % Select the solution with the best fitness
            [~,bestLocal] = min(FitnessAll(Ti));
            x = Ti(bestLocal);

        end

        %% Save selected solution
        selectedCount = selectedCount + 1;

        selectedIndex(selectedCount)   = x;
        selectedFitness(selectedCount) = FitnessAll(x);

        selected(x) = true;
    end

    %% Output
    selectedIndex   = selectedIndex(1:selectedCount);
    selectedFitness = selectedFitness(1:selectedCount);

    Population = Population(selectedIndex);
    Fitness    = selectedFitness;

end
