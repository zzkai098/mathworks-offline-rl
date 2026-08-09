%% Week 10 — cost-aware backtest of the SHIPPED tuned-DQN (fixed 6.1) vs 60/40
% Loads the 5 seed agents trained by scripts/train_dqn_seeds.sh (Polyak tau=1e-3,
% LR 1e-4, ME=100), replays them + a 60/40 baseline through the same 30 hold-out
% windows, charges a drift-aware turnover cost swept over 0/5/10/20 bp, and reports
% the standard metric set with a block-bootstrap Sharpe CI and Deflated Sharpe.
%
% Metric set (per user): Success rate, Mean Sharpe, Across-seed std, Mean MaxDD,
% MaxDD P90, Worst Sharpe, Terminal P10, Mean Terminal.  (Across-seed std / Worst
% Sharpe need >1 seed -> that's why we train 5.)
% Honesty: DSR trial count set to the LARGE true number of configs tried across
% week4-10 (dozens) -> DSR will land near/below the floor. That is the point:
% the contribution is the diagnosis, not a returns edge.

clear; clc;
addpath(fullfile("src","utils"));   % inputTestData, blockBootstrapSharpeCI, deflatedSharpe

%% Fixed params (match training)
initialWealth = 100000;  goalWealth = 102000;  trainingRange = 3020;
horizonPeriods = 30;  numActions = 15;
numEvalEpisodes = 30;  evalStepSize = 30;
seeds = [1000 2000 3000 4000 5000];  nSeeds = numel(seeds);
costBps = [0 5 10 20];  nCost = numel(costBps);
annualizeFactor = sqrt(252);  blockLen = horizonPeriods;  nBoot = 2000;
dsrNumTrials = 40;          % honest: dozens of configs tried across week4-10
dsrTrialSharpeVar = [];     % fallback 1/T if empty

%% Shared basis: frontier + macro z-score from extended train
pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
nAssets = numel(assetNames);
p = Portfolio("AssetList", assetNames);
p = setDefaultConstraints(p);  p = setAssetMoments(p, mu, Sigma);
W_frontier = estimateFrontier(p, numActions);

macroTT = readtable("data/macro_train_extended.csv");
M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);

%% 60/40 static weights (TLT=bond, GLD=gold, rest equity)
upNames = upper(string(assetNames));
isBond = ismember(upNames, ["TLT","IEF","AGG","BND","SHY","LQD"]);
isGold = ismember(upNames, ["GLD","IAU"]);
isEquity = ~isBond & ~isGold;
w6040 = zeros(nAssets,1);
w6040(isEquity) = 0.60/nnz(isEquity);
w6040(isBond)   = 0.40/nnz(isBond);
fprintf("60/40: %d equity / %d bond cols\n", nnz(isEquity), nnz(isBond));

%% Replay each DQN seed + 60/40 -> per-window gross + turnover
dqnGross = cell(nSeeds,1);  dqnTurn = cell(nSeeds,1);
for si = 1:nSeeds
    f = sprintf("experiments/models/dqn_seed_%d.mat", seeds(si));
    assert(isfile(f), "missing %s — run scripts/train_dqn_seeds.sh first", f);
    L = load(f, "FA");  agent = L.FA.agent;
    assert(contains(L.FA.meta.learner, "tuned-DQN"), ...
        "wrong-config agent in %s: %s", f, L.FA.meta.learner);   % provenance guard
    gW = nan(numEvalEpisodes, horizonPeriods);  tW = gW;
    for ep = 1:numEvalEpisodes
        s0 = (ep-1)*evalStepSize + 1;
        if s0 + horizonPeriods - 1 > size(R_full,1); break; end
        [g,tu] = runWindow(agent, 'dqn', W_frontier, [], s0, horizonPeriods, ...
                           R_full, M_test_z, initialWealth, goalWealth);
        gW(ep,:) = g';  tW(ep,:) = tu';
    end
    dqnGross{si} = gW;  dqnTurn{si} = tW;
end

% IQL (optional co-shipped agent — replayed only if its seed bundles exist)
haveIQL = isfile("experiments/models/iql_seed_1000.mat");
iqlGross = cell(nSeeds,1);  iqlTurn = cell(nSeeds,1);
if haveIQL
    for si = 1:nSeeds
        f = sprintf("experiments/models/iql_seed_%d.mat", seeds(si));
        assert(isfile(f), "missing %s — run scripts/train_iql_seeds.sh", f);
        L = load(f, "FI");
        assert(contains(L.FI.meta.learner, "IQL"), "wrong-config bundle in %s", f);
        qNet = L.FI.qNet;
        gW = nan(numEvalEpisodes, horizonPeriods);  tW = gW;
        for ep = 1:numEvalEpisodes
            s0 = (ep-1)*evalStepSize + 1;
            if s0 + horizonPeriods - 1 > size(R_full,1); break; end
            [g,tu] = runWindow(qNet, 'iql', W_frontier, [], s0, horizonPeriods, ...
                               R_full, M_test_z, initialWealth, goalWealth);
            gW(ep,:) = g';  tW(ep,:) = tu';
        end
        iqlGross{si} = gW;  iqlTurn{si} = tW;
    end
end

bG = nan(numEvalEpisodes, horizonPeriods);  bT = bG;
for ep = 1:numEvalEpisodes
    s0 = (ep-1)*evalStepSize + 1;
    if s0 + horizonPeriods - 1 > size(R_full,1); break; end
    [g,tu] = runWindow([], 'static', [], w6040, s0, horizonPeriods, ...
                       R_full, M_test_z, initialWealth, goalWealth);
    bG(ep,:) = g';  bT(ep,:) = tu';
end
fprintf("Replayed 5 DQN seeds + 60/40 across %d windows.\n\n", numEvalEpisodes);

%% Report per cost
if haveIQL; learners = "tuned-DQN + IQL (5 seeds each)"; else; learners = "tuned-DQN (5 seeds)"; end
fprintf("===== Week 10 cost-aware backtest: %s vs 60/40 =====\n", learners);
fprintf("Sharpe annualized; MaxDD/Terminal across windows; CI=95%% block bootstrap; DSR trials=%d\n\n", dsrNumTrials);

for ci = 1:nCost
    c = costBps(ci)/1e4;
    fprintf("--- Transaction cost = %d bp ---\n", costBps(ci));
    fprintf("%-9s | Succ | MeanSh | AcrSd | MeanDD | DD_P90 | WorstSh | TermP10 | MeanTerm | 95%%CI          | DSR\n","strat");

    % tuned-DQN: display metrics aggregate across seeds; CI/DSR on the SHIPPED
    % single agent (seed 1000) only — pooling 5 seeds' returns (same market path)
    % would fabricate independence, inflate T, and splice bootstrap blocks.
    perSeedSharpe = nan(nSeeds,1);  perSeedSucc = nan(nSeeds,1);  perSeedMin = nan(nSeeds,1);
    allDD=[]; allTermW=[]; shipNetR=[];
    for si = 1:nSeeds
        [shW, ddW, twW, scW, ~, netR] = perWindow(dqnGross{si}, dqnTurn{si}, c, initialWealth, goalWealth, annualizeFactor);
        perSeedSharpe(si) = mean(shW,'omitnan');  perSeedSucc(si) = sum(scW,'omitnan');
        perSeedMin(si) = min(shW);                                    % per-seed worst window
        allDD=[allDD; ddW]; allTermW=[allTermW; twW]; %#ok<AGROW>     % descriptive percentiles: pooling OK
        if seeds(si)==1000; shipNetR = netR; end                     % shipped agent
    end
    [lo,hi] = blockBootstrapSharpeCI(shipNetR, blockLen, nBoot, annualizeFactor);
    dsr = deflatedSharpe(shipNetR, dsrNumTrials, dsrTrialSharpeVar);
    % Worst Sharpe = mean across seeds of each seed's worst window (historical semantic)
    fprintf("%-9s | %4.1f | %6.2f | %5.2f | %5.1f%% | %5.1f%% | %7.2f | %7.0f | %8.0f | [%5.2f,%5.2f] | %.3f\n", ...
        "tuned-DQN", mean(perSeedSucc), mean(perSeedSharpe), std(perSeedSharpe), ...
        100*mean(allDD), 100*prctile(allDD,90), mean(perSeedMin), prctile(allTermW,10), mean(allTermW), lo, hi, dsr);

    % IQL: same aggregation (CI/DSR on shipped seed 1000)
    if haveIQL
        pS = nan(nSeeds,1);  pSucc = nan(nSeeds,1);  pMin = nan(nSeeds,1);
        aDD=[]; aTW=[]; shipR=[];
        for si = 1:nSeeds
            [shW, ddW, twW, scW, ~, netR] = perWindow(iqlGross{si}, iqlTurn{si}, c, initialWealth, goalWealth, annualizeFactor);
            pS(si) = mean(shW,'omitnan');  pSucc(si) = sum(scW,'omitnan');  pMin(si) = min(shW);
            aDD=[aDD; ddW]; aTW=[aTW; twW]; %#ok<AGROW>
            if seeds(si)==1000; shipR = netR; end
        end
        [lo2,hi2] = blockBootstrapSharpeCI(shipR, blockLen, nBoot, annualizeFactor);
        dsr2 = deflatedSharpe(shipR, dsrNumTrials, dsrTrialSharpeVar);
        fprintf("%-9s | %4.1f | %6.2f | %5.2f | %5.1f%% | %5.1f%% | %7.2f | %7.0f | %8.0f | [%5.2f,%5.2f] | %.3f\n", ...
            "IQL", mean(pSucc), mean(pS), std(pS), ...
            100*mean(aDD), 100*prctile(aDD,90), mean(pMin), prctile(aTW,10), mean(aTW), lo2, hi2, dsr2);
    end

    % 60/40: single
    [shW, ddW, twW, scW, ~, netR] = perWindow(bG, bT, c, initialWealth, goalWealth, annualizeFactor);
    [lo,hi] = blockBootstrapSharpeCI(netR, blockLen, nBoot, annualizeFactor);
    dsr = deflatedSharpe(netR, dsrNumTrials, dsrTrialSharpeVar);
    fprintf("%-9s | %4.1f | %6.2f | %5s | %5.1f%% | %5.1f%% | %7.2f | %7.0f | %8.0f | [%5.2f,%5.2f] | %.3f\n\n", ...
        "60/40", sum(scW), mean(shW,'omitnan'), "n/a", ...
        100*mean(ddW), 100*prctile(ddW,90), min(shW), prctile(twW,10), mean(twW), lo, hi, dsr);
end

fprintf("Note: CI/DSR are computed on the SHIPPED single agent (seed 1000) return series,\n");
fprintf("NOT pooled across seeds (pooling the same market path fabricates independence).\n");
fprintf("DSR trials=%d (dozens of configs across week4-10); 1/T is a lower bound on\n", dsrNumTrials);
fprintf("cross-trial dispersion, so the true DSR is if anything lower.\n");
fprintf("Reading: with CIs overlapping and DSR near/below floor, the shipped agent is NOT\n");
fprintf("statistically distinguishable from 60/40 net of cost — as expected on ~600\n");
fprintf("single-regime test days. The deliverable is the DIAGNOSIS.\n");


%% ---- local functions (runWindow copied from week8_backtest_harness) ----
function [grossRet, turnover] = runWindow(agent, agentType, basis, staticW, ...
        startEval, H, R_full, M_test_z, initialWealth, goalWealth)
    normalizeWealth = @(w) min(w/goalWealth, 5);
    timeFrac = @(t) t / H;
    grossRet = zeros(H,1);  turnover = zeros(H,1);
    wealth = initialWealth;  wPrevDrift = [];
    for t = 1:H
        dayIdx = startEval + t - 1;
        if isempty(agent)
            w = staticW(:);
        elseif strcmp(agentType, 'iql')
            obs = [normalizeWealth(wealth); timeFrac(t); M_test_z(dayIdx,:)'];
            q = extractdata(forward(agent, dlarray(obs, 'CB')));   % agent = qNet
            [~, a] = max(q);  w = basis(:, a);  w = w(:);
        else   % dqn
            obs = { [normalizeWealth(wealth); timeFrac(t); M_test_z(dayIdx,:)'] };
            aOut = agent.getAction(obs);
            w = basis(:, aOut{1});  w = w(:);
        end
        rAsset = R_full(dayIdx,:)';  rG = R_full(dayIdx,:) * w;
        grossRet(t) = rG;
        if t == 1; turnover(t) = 0; else; turnover(t) = sum(abs(w - wPrevDrift)); end
        wPrevDrift = (w .* (1 + rAsset)) / (1 + rG);
        wealth = wealth * (1 + rG);
    end
end

function [sharpes, maxDDs, termWs, succ, turns, pooledNetRet] = ...
        perWindow(gross, turnover, c, initialWealth, goalWealth, annualizeFactor)
    nWin = size(gross,1);
    sharpes=nan(nWin,1); maxDDs=nan(nWin,1); termWs=nan(nWin,1); succ=nan(nWin,1); turns=nan(nWin,1);
    pooledNetRet = [];
    for ep = 1:nWin
        g = gross(ep,:)'; tu = turnover(ep,:)';
        if any(isnan(g)); continue; end
        netRet = g - c*tu;
        wealth = initialWealth * [1; cumprod(1+netRet)];
        logNet = diff(log(wealth));
        sharpes(ep) = mean(logNet)/std(logNet)*annualizeFactor;
        maxDDs(ep) = max(cummax(wealth)-wealth)/max(wealth);
        termWs(ep) = wealth(end);
        succ(ep) = double(wealth(end) >= goalWealth);
        turns(ep) = mean(tu);
        pooledNetRet = [pooledNetRet; logNet]; %#ok<AGROW>
    end
end
