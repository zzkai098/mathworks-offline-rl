%% Week 10 — 15 vs 20 frontier actions (mentor ask #2): does finer granularity help?
% Compares 15-action agents (dqn_seed_*) vs 20-action agents (dqn20_seed_*) on the
% chained test path (net 10bp): total return per seed + effective-action-count
% (distinct greedy actions used) + how often the argmax lands on adjacent near-twins.
% Expected null: adjacent frontier points are ~collinear, so 20 actions don't change
% realized behavior; per-action offline coverage is thinner -> not better.

clear; clc;
addpath(fullfile("src","utils"));
c = 10/1e4;  seeds = [1000 2000 3000];
initialWealth = 100000;  goalWealth = 102000;
horizonPeriods = 30;  numEvalEpisodes = 30;  evalStepSize = 30;  annualizeFactor = sqrt(252);

macroTT = readtable("data/macro_train_extended.csv");  M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);

fprintf("=== 15 vs 20 frontier actions — test total return (net 10bp) + effA ===\n");
fprintf("%-6s | 15-act: ret / effA | 20-act: ret / effA\n","seed");
r15=nan(3,1); r20=nan(3,1); e15=nan(3,1); e20=nan(3,1);
for i = 1:3
    F15 = load(sprintf("experiments/models/dqn_seed_%d.mat",seeds(i)),"FA").FA;
    F20 = load(sprintf("experiments/models/dqn20_seed_%d.mat",seeds(i)),"FA").FA;
    [r15(i),e15(i)] = evalAgent(F15, numEvalEpisodes, evalStepSize, horizonPeriods, R_full, M_test_z, initialWealth, goalWealth, c);
    [r20(i),e20(i)] = evalAgent(F20, numEvalEpisodes, evalStepSize, horizonPeriods, R_full, M_test_z, initialWealth, goalWealth, c);
    fprintf("%-6d | %+6.1f%% / %2d/15    | %+6.1f%% / %2d/20\n", seeds(i), 100*r15(i), e15(i), 100*r20(i), e20(i));
end
fprintf("\nmean total return: 15-act %+.1f%% | 20-act %+.1f%%\n", 100*mean(r15), 100*mean(r20));
fprintf("mean effective actions used: 15-act %.1f/15 | 20-act %.1f/20\n", mean(e15), mean(e20));
fprintf("(baselines: 60/40 +5.9%% | 1/N +37.4%%)\n");


function [tr, effA] = evalAgent(FA, nEp, step, H, R_full, M_test_z, iw, gw, c)
    nW = @(w) min(w/gw,5);  tF = @(t) t/H;
    Wf = FA.W_frontier;  netChain = [];  usedA = [];
    for ep = 1:nEp
        s0 = (ep-1)*step + 1;  if s0+H-1 > size(R_full,1); break; end
        wealth = iw;  wPrev = [];
        for t = 1:H
            d = s0+t-1;  a = FA.agent.getAction({[nW(wealth); tF(t); M_test_z(d,:)']});
            aI = a{1};  usedA(end+1) = aI; %#ok<AGROW>
            w = Wf(:,aI);  rAsset = R_full(d,:)';  rG = R_full(d,:)*w;
            if t==1; tu=0; else; tu=sum(abs(w-wPrev)); end
            netChain = [netChain; rG - c*tu]; %#ok<AGROW>
            wPrev = (w.*(1+rAsset))/(1+rG);  wealth = wealth*(1+rG);
        end
    end
    wc = iw*[1; cumprod(1+netChain)];  tr = wc(end)/wc(1)-1;  effA = numel(unique(usedA));
end
