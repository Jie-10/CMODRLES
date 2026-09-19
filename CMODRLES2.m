classdef CMODRLES2 < ALGORITHM
% <2026> <multi/many> <real> <constrained/none>
% Deep reinforcement learning assisted auxiliary-task selection
%
% The evolutionary framework contains three populations:
% MP : constraint-oriented main population (SPEA2-CDP)
% AP : objective/diversity-oriented auxiliary population
% DP : constraint/diversity-oriented auxiliary population
%
% Action 1 selects AP to generate auxiliary offspring.
% Action 2 selects DP to generate auxiliary offspring.
%
% The DRL implementation follows the coding style of DRLOS-EMCMO in
% PlatEMO: Data replay, random exploration, trainmodel/testNet/updatemodel,
% and periodic Q-network updating are all managed directly in main().

    methods
        function main(Algorithm,Problem)
            %% Initialize MP, AP and DP
            Population{1} = Problem.Initialization();
            Population{2} = Population{1};
            Population{3} = Population{1};
            Fitness{1}    = CalFitness(Population{1}.objs,Population{1}.cons,0);
            Fitness{2}    = CalFitness(Population{2}.objs);
            Fitness{3}    = Fitness{1};

            N = length(Population{1});
            [W,~] = UniformPoint(N,Problem.M);

            %% Reference range used by the second state variable
            InitObj = [Population{1}.objs;Population{2}.objs];
            Ref.lo = min(InitObj,[],1);
            Ref.hi = max(InitObj,[],1);
            Ref.spanFloor = 1e-12*max(1,max(abs(InitObj),[],1));

            %% Calculate the minimum angular threshold
            CosWW = 1-pdist2(W,W,'cosine');
            CosWW = min(max(CosWW,-1),1);
            AngleW = acos(CosWW);
            AngleW(logical(eye(size(AngleW,1)))) = inf;
            if size(W,1) > 1
                MinAngle = mean(min(AngleW,[],2))/2;
            else
                MinAngle = pi/2;
            end

            %% For DQL (same organization as DRLOS-EMCMO)
            Data         = [];
            num_action   = 2;
            model_built  = 0;
            count        = 0;
            greedy       = 0.9;
            gama         = 0.9;

            %% Optimization
            while Algorithm.NotTerminated(Population{1})
                gen = ceil(Problem.FE/(2*Problem.N));

                %% Old state
                state = CMODRLES_State(Population{1}.objs,Population{1}.cons,W,Ref);

                if gen <= 200
                    %% Exploring stage: choose AP or DP randomly
                    action = randi(num_action);
                else
                    %% Learning stage: choose AP or DP by the Q-network
                    if ~model_built
                        % Build the model after 200 decision records
                        use_data = randperm(size(Data,1),200);
                        tr_x = Data(use_data,1:3);
                        [tr_xx,ps] = mapminmax(tr_x');
                        tr_xx = tr_xx';
                        tr_y = Data(use_data,4:6);
                        [tr_yy,qs] = mapminmax(tr_y');
                        tr_yy = tr_yy';

                        Params.ps = ps;
                        Params.qs = qs;
                        [net,Params] = trainmodel(tr_xx,tr_yy,Params);
                        model_built = 1;
                        action = randi(num_action);
                    else
                        % Choose an action according to the trained network
                        if rand > greedy
                            action = randi(num_action);
                        else
                            test_x1 = [state,1];
                            test_x2 = [state,2];
                            ps = Params.ps;
                            qs = Params.qs;

                            x1 = mapminmax('apply',test_x1',ps);
                            x1 = x1';
                            x2 = mapminmax('apply',test_x2',ps);
                            x2 = x2';

                            q1 = testNet(x1,net,Params);
                            q1 = mapminmax('reverse',q1',qs);
                            q1 = q1';
                            q2 = testNet(x2,net,Params);
                            q2 = mapminmax('reverse',q2',qs);
                            q2 = q2';

                            [~,action] = max([q1(1);q2(1)]);
                        end
                    end
                end

                %% Parent selection and offspring generation
                Zmin = min([Population{1}.objs;Population{2}.objs;Population{3}.objs],[],1);
                auxIndex = action + 1;

                MatingPool1 = TournamentSelection(2,N,Fitness{1});
                MatingPool2 = TournamentSelection(2,N,Fitness{auxIndex});

                if rand > 0.5
                    Offspring{1} = Neighbor_Pairing_Strategy(Problem, ...
                        Population{1}(MatingPool1),Population{1},Zmin);
                    Offspring{2} = Neighbor_Pairing_Strategy(Problem, ...
                        Population{auxIndex}(MatingPool2),Population{auxIndex},Zmin);
                else
                    Offspring{1} = OperatorDE(Problem,Population{1}, ...
                        Population{1}(randperm(N)),Population{1}(randperm(N)));
                    Offspring{2} = OperatorDE(Problem,Population{auxIndex}, ...
                        Population{auxIndex}(randperm(N)),Population{auxIndex}(randperm(N)));
                end

                %% Environmental selection
                Shared = [Offspring{1},Offspring{2}];
                CandidateMP = [Population{1},Shared];

                % MP: SPEA2-CDP
                [Population{1},Fitness{1},selectedIndex] = ...
                    EnviromentSelect1(CandidateMP,N);

                % AP: diversity-oriented SPEA2 without constraints
                [Population{2},Fitness{2}] = EnviromentSelect2( ...
                    [Population{2},Shared],N,MinAngle,W);

                % DP: diversity-oriented SPEA2-CDP
                [Population{3},Fitness{3}] = EnviromentSelect3( ...
                    [Population{3},Shared],N,MinAngle,W);

                %% Update the observed objective range and calculate new state
                Ref.lo = min(Ref.lo,min(Shared.objs,[],1));
                Ref.hi = max(Ref.hi,max(Shared.objs,[],1));
                next_state = CMODRLES_State(Population{1}.objs,Population{1}.cons,W,Ref);

                %% Update experience replay
                reward = CMODRLES_Reward(CandidateMP,selectedIndex,N,N,N);
                current_record = [state,action,reward,next_state];
                Data = [Data;current_record];
                if size(Data,1) > 500
                    Data(1,:) = [];
                end

                %% Update Q-network
                % Update the network every 50 generations after it is built
                if model_built
                    count = count + 1;
                    if count > 50
                        use_data = randperm(size(Data,1),200);
                        tr_x = Data(use_data,1:3);
                        [tr_xx,ps] = mapminmax(tr_x');
                        tr_xx = tr_xx';

                        reward_est = testNet(tr_xx,net,Params);
                        qs = Params.qs;
                        reward_est = mapminmax('reverse',reward_est',qs);
                        reward_est = reward_est';
                        succ = reward_est(:,1);

                        tr_yy = Data(use_data,4) + gama*max(succ);
                        [tr_yy,qs] = mapminmax(tr_yy');
                        tr_yy = tr_yy';
                        Params.ps = ps;
                        Params.qs = qs;
                        net = updatemodel(tr_xx,tr_yy,Params,net);
                        count = 0;
                    end
                end
            end
        end
    end
end
