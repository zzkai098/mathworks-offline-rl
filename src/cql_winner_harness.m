%% CQL winner — cost-aware backtest (CQL α=1 vs DDPG anchor vs 60/40)
% Runs the sweep WINNER (CQL-H, cqlAlpha=1) through the same cost-aware,
% statistically-honest harness as Week 8, against two references:
%   - DDPG anchor (cqlAlpha=0, SAME hand-written trainer) — isolates the CQL
%     effect apples-to-apples (identical pipeline, only the penalty differs)
%   - 60/40 static daily constant-mix baseline
%
% Honest framing (pre-registered): the CQL win is EFFECT ① — it cures the DDPG
% seed collapse and is the most reproducible across seeds. It is NOT expected to
% be statistically distinguishable from 60/40 on net Sharpe (Week 8 showed all
% CIs overlap 0 in this single test regime). This harness reports that honestly:
% net Sharpe + 95% block-bootstrap CI + Deflated Sharpe with a REAL multiple-
% testing haircut (trial count + trial-Sharpe variance from the observed spread).
%
% Reuses src/utils: inputTestData, blockBootstrapSharpeCI, deflatedSharpe.

clear; clc;
addpath(fullfile("src", "utils"));

%% Fixed parameters (match the training scripts)
initialWealth   = 100000;
goalWealth      = 102000;
trainingRange   = 3020;
horizonPeriods  = 30;
numActions      = 15;
nDense          = 200;
numEvalEpisodes = 30;
evalStepSize    = 30;
seeds           = [1000 2000 3000 4000 5000];
nSeeds          = numel(seeds);

costBps         = [0 5 10 20];
nCost           = numel(costBps);
annualizeFactor = sqrt(252);
blockLen        = horizonPeriods;
nBoot           = 2000;

% --- Honest DSR haircut (reviewer SHOULD-FIX 3) ---
% Trial-Sharpe variance MUST be estimated over the FULL trial universe actually
% searched — including the LOSING configs — otherwise the haircut is fit on a
% favorable subset and DSR is inflated. These are the config-level mean Sharpes
% across Weeks 4-8 + the CQL sweep (from CLAUDE.md experiment log), the bad ones
% included.
trialSharpesAnnual = [ ...
    0.59 -0.01 0.25 ...          % Week4 baseline / v2 / v3
    0.37 0.50 0.36 ...           % Week5 v1 / v2 / PlanC
    0.62 0.55 0.51 ...           % Week6.0 / 6.0b-s1 / 6.0b-s2
    0.66 0.54 0.57 ...           % Week6.1 / 7.0 / 7.0b
    0.29 0.35 ...                % Week7 DDPG / Week8 DDPG on-frontier
    0.24 0.70 0.78 ];            % CQL sweep a=0.3 / a=1 / a=3
dsrNumTrials      = numel(trialSharpesAnnual);          % honest trial count = 17
dsrTrialSharpeVar = var(trialSharpesAnnual) / 252;      % per-period
fprintf("DSR haircut (honest): %d trials, trial-Sharpe var (per-period) = %.3e\n", ...
    dsrNumTrials, dsrTrialSharpeVar);

logRootCQL  = fullfile("experiments","logs","cql_ddpg","sweep_alpha1");
logRootDDPG = fullfile("experiments","logs","cql_ddpg","alpha_0");
logRoot61   = fullfile("experiments","logs","week6_1_regime_gated");

%% Shared basis: dense frontier + macro z-score from the extended train
pricesTT   = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train    = pricesTT{:, 2:end};
R_train    = diff(log(P_train));
R_sub      = R_train(1:trainingRange, :);
mu         = mean(R_sub, 1)';
Sigma      = cov(R_sub);
nAssets    = numel(assetNames);

p = Portfolio("AssetList", assetNames);
p = setDefaultConstraints(p);
p = setAssetMoments(p, mu, Sigma);
W_frontier = estimateFrontier(p, numActions);
rmin    = estimatePortReturn(p, W_frontier(:, 1));
rmax    = estimatePortReturn(p, W_frontier(:, end));
W_dense = estimateFrontierByReturn(p, linspace(rmin, rmax, nDense));

macroTT   = readtable("data/macro_train_extended.csv");
M_train   = macroTT{:, 2:end};
macroMean = mean(M_train, 1);
macroStd  = std(M_train, 0, 1);
macroStd(macroStd == 0) = 1;

[R_full, M_test_z] = inputTestData('data/prices_test.csv', 'data/macro_test.csv', ...
                                   macroMean, macroStd);

%% 60/40 static target weights
bondNames = {'TLT','IEF','AGG','BND','SHY','LQD'};
goldNames = {'GLD','IAU'};
upNames   = upper(string(assetNames));
isBond    = ismember(upNames, upper(string(bondNames)));
isGold    = ismember(upNames, upper(string(goldNames)));
isEquity  = ~isBond & ~isGold;
w6040 = zeros(nAssets, 1);
w6040(isEquity) = 0.60 / nnz(isEquity);
w6040(isBond)   = 0.40 / nnz(isBond);
fprintf("60/40 baseline: %d equity (60%%), %d bond (40%%), %d gold (0%%)\n", ...
    nnz(isEquity), nnz(isBond), nnz(isGold));

%% Collect per-window gross returns + turnover for each RL strategy
% CQL a=1 and DDPG a=0 are continuous (ddpg, W_dense); 6.1 is discrete
% (dqn, 15 W_frontier portfolios).
grossRL = struct('name',      {'CQL a=1',   'DDPG a=0',  '6.1 DQN'}, ...
                 'agentType', {'ddpg',      'ddpg',      'dqn'}, ...
                 'basis',     {W_dense,     W_dense,     W_frontier}, ...
                 'logRoot',   {logRootCQL,  logRootDDPG, logRoot61});
for s = 1:numel(grossRL)
    grossRL(s).gross    = cell(nSeeds, 1);
    grossRL(s).turnover = cell(nSeeds, 1);
    for si = 1:nSeeds
        agentFile = fullfile(grossRL(s).logRoot, sprintf("seed_%d", seeds(si)), "TrainedAgent.mat");
        if ~isfile(agentFile)
            error("Missing trained agent: %s", agentFile);
        end
        L = load(agentFile, "agent"); agent = L.agent;
        gW = nan(numEvalEpisodes, horizonPeriods);
        tW = nan(numEvalEpisodes, horizonPeriods);
        for ep = 1:numEvalEpisodes
            startEval = (ep-1)*evalStepSize + 1;
            if startEval + horizonPeriods - 1 > size(R_full, 1); break; end
            [g, t] = runWindow(agent, grossRL(s).agentType, grossRL(s).basis, [], ...
                               startEval, horizonPeriods, ...
                               R_full, M_test_z, initialWealth, goalWealth);
            gW(ep, :) = g'; tW(ep, :) = t';
        end
        grossRL(s).gross{si} = gW; grossRL(s).turnover{si} = tW;
    end
    fprintf("Replayed %s across %d seeds.\n", grossRL(s).name, nSeeds);
end

% 60/40 baseline
gB = nan(numEvalEpisodes, horizonPeriods); tB = nan(numEvalEpisodes, horizonPeriods);
for ep = 1:numEvalEpisodes
    startEval = (ep-1)*evalStepSize + 1;
    if startEval + horizonPeriods - 1 > size(R_full, 1); break; end
    [g, t] = runWindow([], 'static', [], w6040, startEval, horizonPeriods, ...
                       R_full, M_test_z, initialWealth, goalWealth);
    gB(ep, :) = g'; tB(ep, :) = t';
end
fprintf("Ran 60/40 baseline.\n\n");

%% Metrics per (strategy x cost)
nRL        = numel(grossRL);
stratNames = [{grossRL.name}, {'60/40'}];
nStrat     = numel(stratNames);
netSharpe = nan(nStrat,nCost); netSharpeSd = nan(nStrat,nCost);
ciLow = nan(nStrat,nCost); ciHigh = nan(nStrat,nCost);
maxDDp90 = nan(nStrat,nCost); termP10 = nan(nStrat,nCost);
meanTurn = nan(nStrat,nCost); succRate = nan(nStrat,nCost); dsrAtCost = nan(nStrat,nCost);

% Estimator discipline (reviewer B1/B2): the point Sharpe, its CI, and the DSR
% are ALL the POOLED-DAILY Sharpe on the same net-return series — never the
% window-averaged Sharpe paired with a pooled-daily CI. For RL strategies we do
% NOT concatenate the 5 seeds into one series (that inflates T and mechanically
% tightens the CI); instead each seed is bootstrapped on its own ~900 daily
% returns and the per-seed point/CI/DSR are averaged across seeds. Reproducibility
% lives in the across-seed std column, not smuggled into the CI.
poolSharpe = @(r) mean(r) / std(r) * annualizeFactor;   % pooled-daily estimator

for ci = 1:nCost
    c = costBps(ci) / 1e4;
    for s = 1:nStrat
        if s <= nRL
            perSeedSharpe = nan(nSeeds,1); perSeedLo = nan(nSeeds,1); perSeedHi = nan(nSeeds,1);
            perSeedDSR = nan(nSeeds,1); perSeedP90 = nan(nSeeds,1); perSeedP10 = nan(nSeeds,1);
            perSeedSucc = nan(nSeeds,1); perSeedTurn = nan(nSeeds,1);
            for si = 1:nSeeds
                [~,p90,p10,suc,trn,netRets] = windowStats(grossRL(s).gross{si}, ...
                    grossRL(s).turnover{si}, c, initialWealth, goalWealth, annualizeFactor);
                perSeedSharpe(si) = poolSharpe(netRets);          % pooled-daily, this seed
                [lo,hi] = blockBootstrapSharpeCI(netRets, blockLen, nBoot, annualizeFactor);
                perSeedLo(si) = lo; perSeedHi(si) = hi;
                perSeedDSR(si) = deflatedSharpe(netRets, dsrNumTrials, dsrTrialSharpeVar);
                perSeedP90(si)=p90; perSeedP10(si)=p10; perSeedSucc(si)=suc; perSeedTurn(si)=trn;
            end
            netSharpe(s,ci)=mean(perSeedSharpe,'omitnan'); netSharpeSd(s,ci)=std(perSeedSharpe,'omitnan');
            ciLow(s,ci)=mean(perSeedLo,'omitnan'); ciHigh(s,ci)=mean(perSeedHi,'omitnan');
            dsrAtCost(s,ci)=mean(perSeedDSR,'omitnan');
            maxDDp90(s,ci)=mean(perSeedP90,'omitnan'); termP10(s,ci)=mean(perSeedP10,'omitnan');
            succRate(s,ci)=mean(perSeedSucc,'omitnan'); meanTurn(s,ci)=mean(perSeedTurn,'omitnan');
        else
            [~,p90,p10,suc,trn,netRets] = windowStats(gB,tB,c, ...
                initialWealth,goalWealth,annualizeFactor);
            netSharpe(s,ci)=poolSharpe(netRets);                  % pooled-daily, single series
            [lo,hi] = blockBootstrapSharpeCI(netRets, blockLen, nBoot, annualizeFactor);
            ciLow(s,ci)=lo; ciHigh(s,ci)=hi;
            dsrAtCost(s,ci)=deflatedSharpe(netRets, dsrNumTrials, dsrTrialSharpeVar);
            maxDDp90(s,ci)=p90; termP10(s,ci)=p10; succRate(s,ci)=suc; meanTurn(s,ci)=trn;
        end
    end
end

%% Report
fprintf("\n===== CQL winner cost-aware backtest (30 windows, daily, lane B) =====\n\n");
for ci = 1:nCost
    fprintf("--- Transaction cost = %d bp ---\n", costBps(ci));
    fprintf("%-9s | netSharpe (95%% CI)      | MaxDD P90 | TermP10 | turn/step | succ | DSR\n","strat");
    for s = 1:nStrat
        sdTxt = ""; if s <= nRL, sdTxt = sprintf("±%.2f", netSharpeSd(s,ci)); end
        fprintf("%-9s | %5.2f %-5s [%5.2f,%5.2f] | %7.2f%% | %7.0f | %8.4f | %4.1f | %.3f\n", ...
            stratNames{s}, netSharpe(s,ci), sdTxt, ciLow(s,ci), ciHigh(s,ci), ...
            maxDDp90(s,ci)*100, termP10(s,ci), meanTurn(s,ci), succRate(s,ci), dsrAtCost(s,ci));
    end
    fprintf("\n");
end

save(fullfile("experiments","logs","cql_winner_backtest.mat"), ...
    "stratNames","costBps","netSharpe","netSharpeSd","ciLow","ciHigh", ...
    "maxDDp90","termP10","meanTurn","succRate","dsrAtCost","dsrNumTrials","dsrTrialSharpeVar");

%% ---- Local functions ----
function [grossRet, turnover] = runWindow(agent, agentType, basis, staticW, ...
        startEval, H, R_full, M_test_z, initialWealth, goalWealth)
    normalizeWealth = @(w) min(w/goalWealth, 5);
    timeFrac = @(t) t / H;
    grossRet = zeros(H,1); turnover = zeros(H,1);
    wealth = initialWealth; wPrevDrift = [];
    for t = 1:H
        dayIdx = startEval + t - 1;
        if isempty(agent)
            w = staticW(:);
        else
            obs = { [normalizeWealth(wealth); timeFrac(t); M_test_z(dayIdx,:)'] };
            aOut = agent.getAction(obs);
            switch agentType
                case 'dqn'
                    w = basis(:, aOut{1});
                case 'ddpg'
                    alpha = max(0, min(1, double(aOut{1})));
                    w = alphaToWeights(alpha, basis);
                otherwise
                    error("unknown agentType %s", agentType);
            end
            w = w(:);
        end
        rAsset = R_full(dayIdx,:)'; rG = R_full(dayIdx,:) * w;
        grossRet(t) = rG;
        if t == 1, turnover(t) = 0; else, turnover(t) = sum(abs(w - wPrevDrift)); end
        wPrevDrift = (w .* (1 + rAsset)) / (1 + rG);
        wealth = wealth * (1 + rG);
    end
end

function [meanSharpe,p90MaxDD,p10TermW,succRate,meanTurn,pooledNetRet] = ...
        windowStats(gross, turnover, c, initialWealth, goalWealth, annualizeFactor)
    nWin = size(gross,1);
    sharpes=nan(nWin,1); maxDDs=nan(nWin,1); termWs=nan(nWin,1); succ=nan(nWin,1); turns=nan(nWin,1);
    pooledNetRet = [];
    for ep = 1:nWin
        g = gross(ep,:)'; tu = turnover(ep,:)';
        if any(isnan(g)); continue; end
        netRet = g - c*tu;
        wealth = initialWealth * [1; cumprod(1 + netRet)];
        logNet = diff(log(wealth));
        sharpes(ep) = mean(logNet)/std(logNet)*annualizeFactor;
        maxDDs(ep)  = max(cummax(wealth) - wealth) / max(wealth);
        termWs(ep)  = wealth(end);
        succ(ep)    = double(wealth(end) >= goalWealth);
        turns(ep)   = mean(tu);
        pooledNetRet = [pooledNetRet; logNet]; %#ok<AGROW>
    end
    meanSharpe = mean(sharpes,'omitnan'); p90MaxDD = prctile(maxDDs,90);
    p10TermW = prctile(termWs,10); succRate = sum(succ,'omitnan'); meanTurn = mean(turns,'omitnan');
end

function w = alphaToWeights(alpha, W_basis)
    alpha = max(0, min(1, alpha));
    nCol = size(W_basis,2);
    idxF = 1 + alpha*(nCol-1);
    iLo = max(1, floor(idxF)); iHi = min(nCol, iLo+1); frac = idxF - iLo;
    w = (1-frac)*W_basis(:,iLo) + frac*W_basis(:,iHi);
end
