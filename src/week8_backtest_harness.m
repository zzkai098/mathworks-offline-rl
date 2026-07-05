%% Week 8 — Cost-Aware Backtest Harness (lane B: tactical / risk-adjusted)
% Replays the two locked RL strategies through the SAME hold-out windows as
% their training scripts, adds a 60/40 daily constant-mix baseline, charges
% a turnover-based transaction cost (swept over several levels), and reports
% net risk-adjusted metrics with a block-bootstrap Sharpe CI and a Deflated
% Sharpe. This answers "does dynamic beat static, net of cost?" and is the
% prerequisite for CQL.
%
% Strategies (both trained on the EXTENDED train 2010-2021, so they share
% one frontier basis and one macro z-score — only the action map differs):
%   - 6.1   : discrete DQN over 15 frontier portfolios   (basis = W_frontier)
%   - DDPG  : continuous alpha, ON-FRONTIER dense grid     (basis = W_dense)
%
% Baseline:
%   - 60/40 : daily constant-mix, rebalanced to target each step (pays cost)
%
% Requires the trained agents from prior runs (gitignored):
%   experiments/logs/week6_1_regime_gated/seed_*/TrainedAgent.mat
%   experiments/logs/week8_ddpg_onfrontier/seed_*/TrainedAgent.mat
% Run week6_1_regime_gated.m and week8_ddpg_onfrontier.m first if absent.

clear; clc;
addpath(fullfile("src", "utils"));    % inputTestData, blockBootstrapSharpeCI, deflatedSharpe


%% Fixed parameters (match the training scripts exactly)
initialWealth   = 100000;
goalWealth      = 102000;
trainingRange   = 3020;
horizonPeriods  = 30;                 % H
numActions      = 15;
nDense          = 200;

numEvalEpisodes = 30;
evalStepSize    = 30;
seeds           = [1000 2000 3000 4000 5000];
nSeeds          = numel(seeds);

% Cost sweep (bp of turnover) and stats config
costBps         = [0 5 10 20];
nCost           = numel(costBps);
annualizeFactor = sqrt(252);
blockLen        = horizonPeriods;     % block bootstrap block ~ horizon
nBoot           = 2000;
% DSR inputs — set from the full experiment log for a rigorous number.
% Default: ~10 distinct reward/algorithm configurations were tried.
dsrNumTrials      = 10;
% Variance of the per-period (non-annualized) Sharpe estimates ACROSS those
% trials. [] uses the single-strategy fallback 1/T (then DSR ~ PSR, weakly
% deflated). For a rigorous haircut, compute Var of the trial Sharpes from
% docs/weekly_reports and set it here (remember: per-period = annualized/sqrt(252)).
dsrTrialSharpeVar = [];

logRoot61       = fullfile("experiments", "logs", "week6_1_regime_gated");
logRootDDPG     = fullfile("experiments", "logs", "week8_ddpg_onfrontier");


%% Shared basis: frontier + macro z-score from the extended train
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
W_frontier = estimateFrontier(p, numActions);                 % 6.1 basis (15 pts)

rmin    = estimatePortReturn(p, W_frontier(:, 1));
rmax    = estimatePortReturn(p, W_frontier(:, end));
W_dense = estimateFrontierByReturn(p, linspace(rmin, rmax, nDense));  % DDPG basis

macroTT   = readtable("data/macro_train_extended.csv");
M_train   = macroTT{:, 2:end};
macroMean = mean(M_train, 1);
macroStd  = std(M_train, 0, 1);
macroStd(macroStd == 0) = 1;

% Test data (same helper the training scripts used)
[R_full, M_test_z] = inputTestData('data/prices_test.csv', 'data/macro_test.csv', ...
                                   macroMean, macroStd);


%% 60/40 static target weights (classify columns; override here if needed)
bondNames = {'TLT','IEF','AGG','BND','SHY','LQD'};   % <- edit if your tickers differ
goldNames = {'GLD','IAU'};
upNames   = upper(string(assetNames));
isBond    = ismember(upNames, upper(string(bondNames)));
isGold    = ismember(upNames, upper(string(goldNames)));
isEquity  = ~isBond & ~isGold;
if ~any(isBond) || ~any(isEquity)
    disp("Asset names: " + strjoin(string(assetNames), ", "));
    error(['Could not classify equity/bond for the 60/40 baseline. ', ...
           'Edit bondNames/goldNames at the top of the harness.']);
end
w6040 = zeros(nAssets, 1);
w6040(isEquity) = 0.60 / nnz(isEquity);
w6040(isBond)   = 0.40 / nnz(isBond);
fprintf("60/40 baseline: %d equity cols (60%%), %d bond cols (40%%), %d gold cols (0%%)\n", ...
    nnz(isEquity), nnz(isBond), nnz(isGold));


%% Collect per-window gross returns + turnover (cost-independent) for each strategy
% RL strategies: cell{seed} -> [numEvalEpisodes x horizonPeriods] gross / turnover
grossRL = struct('name', {'6.1', 'DDPG'}, ...
                 'agentType', {'dqn', 'ddpg'}, ...
                 'logRoot', {logRoot61, logRootDDPG}, ...
                 'basis', {W_frontier, W_dense});

for s = 1:numel(grossRL)
    grossRL(s).gross    = cell(nSeeds, 1);
    grossRL(s).turnover = cell(nSeeds, 1);
    for si = 1:nSeeds
        agentFile = fullfile(grossRL(s).logRoot, sprintf("seed_%d", seeds(si)), "TrainedAgent.mat");
        if ~isfile(agentFile)
            error("Missing trained agent: %s (run the training script first).", agentFile);
        end
        L = load(agentFile, "agent");
        agent = L.agent;

        gW = nan(numEvalEpisodes, horizonPeriods);
        tW = nan(numEvalEpisodes, horizonPeriods);
        for ep = 1:numEvalEpisodes
            startEval = (ep-1)*evalStepSize + 1;
            if startEval + horizonPeriods - 1 > size(R_full, 1); break; end
            [g, t] = runWindow(agent, grossRL(s).agentType, grossRL(s).basis, [], ...
                               startEval, horizonPeriods, R_full, M_test_z, ...
                               initialWealth, goalWealth);
            gW(ep, :) = g'; tW(ep, :) = t';
        end
        grossRL(s).gross{si}    = gW;
        grossRL(s).turnover{si} = tW;
    end
    fprintf("Replayed %s across %d seeds.\n", grossRL(s).name, nSeeds);
end

% Baseline 60/40 (single deterministic run)
gB = nan(numEvalEpisodes, horizonPeriods);
tB = nan(numEvalEpisodes, horizonPeriods);
for ep = 1:numEvalEpisodes
    startEval = (ep-1)*evalStepSize + 1;
    if startEval + horizonPeriods - 1 > size(R_full, 1); break; end
    [g, t] = runWindow([], 'static', [], w6040, startEval, horizonPeriods, ...
                       R_full, M_test_z, initialWealth, goalWealth);
    gB(ep, :) = g'; tB(ep, :) = t';
end
fprintf("Ran 60/40 baseline.\n\n");


%% Metrics per (strategy x cost)
stratNames = {grossRL(1).name, grossRL(2).name, '60/40'};
nStrat     = numel(stratNames);

% Result tables: rows = strategy, cols = cost
netSharpe   = nan(nStrat, nCost);   % across-seed mean of per-seed mean per-window Sharpe
netSharpeSd = nan(nStrat, nCost);   % across-seed std (RL only)
ciLow       = nan(nStrat, nCost);
ciHigh      = nan(nStrat, nCost);
maxDDp90    = nan(nStrat, nCost);
termP10     = nan(nStrat, nCost);
meanTurn    = nan(nStrat, nCost);
succRate    = nan(nStrat, nCost);
dsrAtCost   = nan(nStrat, nCost);

for ci = 1:nCost
    c = costBps(ci) / 1e4;
    for s = 1:nStrat
        if s <= 2   % RL strategy (has seeds)
            perSeedSharpe = nan(nSeeds, 1);
            perSeedP90DD  = nan(nSeeds, 1);
            perSeedP10TW  = nan(nSeeds, 1);
            perSeedSucc   = nan(nSeeds, 1);
            perSeedTurn   = nan(nSeeds, 1);
            pooledNetRet  = [];
            for si = 1:nSeeds
                gW = grossRL(s).gross{si};
                tW = grossRL(s).turnover{si};
                [shp, p90, p10, suc, trn, netRets] = ...
                    windowStats(gW, tW, c, initialWealth, goalWealth, annualizeFactor);
                perSeedSharpe(si) = shp;
                perSeedP90DD(si)  = p90;
                perSeedP10TW(si)  = p10;
                perSeedSucc(si)   = suc;
                perSeedTurn(si)   = trn;
                pooledNetRet = [pooledNetRet; netRets]; %#ok<AGROW>
            end
            netSharpe(s, ci)   = mean(perSeedSharpe, 'omitnan');
            netSharpeSd(s, ci) = std(perSeedSharpe, 'omitnan');
            maxDDp90(s, ci)    = mean(perSeedP90DD, 'omitnan');
            termP10(s, ci)     = mean(perSeedP10TW, 'omitnan');
            succRate(s, ci)    = mean(perSeedSucc, 'omitnan');
            meanTurn(s, ci)    = mean(perSeedTurn, 'omitnan');
        else        % 60/40 baseline (no seeds)
            [shp, p90, p10, suc, trn, pooledNetRet] = ...
                windowStats(gB, tB, c, initialWealth, goalWealth, annualizeFactor);
            netSharpe(s, ci) = shp;
            maxDDp90(s, ci)  = p90;
            termP10(s, ci)   = p10;
            succRate(s, ci)  = suc;
            meanTurn(s, ci)  = trn;
        end

        % Block-bootstrap Sharpe CI + Deflated Sharpe on the pooled net returns
        [lo, hi] = blockBootstrapSharpeCI(pooledNetRet, blockLen, nBoot, annualizeFactor);
        ciLow(s, ci)  = lo;
        ciHigh(s, ci) = hi;
        dsrAtCost(s, ci) = deflatedSharpe(pooledNetRet, dsrNumTrials, dsrTrialSharpeVar);
    end
end


%% Report
fprintf("\n===== Week 8 cost-aware backtest (30 windows, daily rebalance, lane B) =====\n");
fprintf("Sharpe annualized x sqrt(252); MaxDD P90 / Terminal P10 across windows;\n");
fprintf("turnover = mean per-step sum|dw| (drift-aware); CI = 95%% block bootstrap.\n\n");

for ci = 1:nCost
    fprintf("--- Transaction cost = %d bp ---\n", costBps(ci));
    fprintf("%-7s | netSharpe (95%% CI)      | MaxDD P90 | TermP10 | turn/step | succ | DSR\n", "strat");
    for s = 1:nStrat
        sdTxt = "";
        if s <= 2, sdTxt = sprintf("±%.2f", netSharpeSd(s, ci)); end
        fprintf("%-7s | %5.2f %-5s [%5.2f,%5.2f] | %7.2f%% | %7.0f | %8.4f | %4.1f | %.3f\n", ...
            stratNames{s}, netSharpe(s, ci), sdTxt, ciLow(s, ci), ciHigh(s, ci), ...
            maxDDp90(s, ci)*100, termP10(s, ci), meanTurn(s, ci), succRate(s, ci), ...
            dsrAtCost(s, ci));
    end
    fprintf("\n");
end

% Headline read at 10 bp
ci10 = find(costBps == 10, 1);
if ~isempty(ci10)
    [~, best] = max(netSharpe(:, ci10));
    fprintf("Headline @10bp: best net Sharpe = %s (%.2f). ", stratNames{best}, netSharpe(best, ci10));
    if best == nStrat
        fprintf("60/40 beats both RL strategies net of cost.\n");
    else
        fprintf("An RL strategy still leads net of cost.\n");
    end
end


%% Save results + figure
if ~isfolder(fullfile("experiments","logs"));    mkdir(fullfile("experiments","logs")); end
if ~isfolder(fullfile("experiments","figures")); mkdir(fullfile("experiments","figures")); end
save(fullfile("experiments","logs","week8_backtest_summary.mat"), ...
    "stratNames", "costBps", "netSharpe", "netSharpeSd", "ciLow", "ciHigh", ...
    "maxDDp90", "termP10", "meanTurn", "succRate", "dsrAtCost");

figure('Name','Week 8 — net Sharpe vs cost');
plot(costBps, netSharpe', '-o', 'LineWidth', 1.8); grid on;
xlabel('Transaction cost (bp of turnover)'); ylabel('Net Sharpe (annualized)');
title('Net Sharpe vs cost — 6.1 / DDPG / 60/40'); legend(stratNames, 'Location','best');
saveas(gcf, fullfile("experiments","figures","week8_net_sharpe.png"));


%% ---- Local functions ----
function [grossRet, turnover] = runWindow(agent, agentType, basis, staticW, ...
        startEval, H, R_full, M_test_z, initialWealth, goalWealth)
    % One eval window. Returns per-step gross return and drift-aware turnover.
    % Portfolio return uses R_full(day,:)*w as in the training scripts (log
    % returns treated as simple, kept identical for reproducibility).
    normalizeWealth = @(w) min(w/goalWealth, 5);
    timeFrac        = @(t) t / H;

    grossRet = zeros(H, 1);
    turnover = zeros(H, 1);
    wealth   = initialWealth;
    wPrevDrift = [];

    for t = 1:H
        dayIdx = startEval + t - 1;
        if isempty(agent)
            w = staticW(:);                              % static baseline target
        else
            macroVec = M_test_z(dayIdx, :)';
            obs = { [normalizeWealth(wealth); timeFrac(t); macroVec] };
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

        rAsset = R_full(dayIdx, :)';                     % asset returns this step
        rG     = R_full(dayIdx, :) * w;                  % portfolio gross return
        grossRet(t) = rG;

        if t == 1
            turnover(t) = 0;                             % entry turnover off
        else
            turnover(t) = sum(abs(w - wPrevDrift));
        end
        % drift current weights over this step for next step's turnover
        wPrevDrift = (w .* (1 + rAsset)) / (1 + rG);
        wealth = wealth * (1 + rG);
    end
end

function [meanSharpe, p90MaxDD, p10TermW, succRate, meanTurn, pooledNetRet] = ...
        windowStats(gross, turnover, c, initialWealth, goalWealth, annualizeFactor)
    % Aggregate per-window net metrics for one strategy at cost c (fraction).
    nWin = size(gross, 1);
    sharpes = nan(nWin, 1);
    maxDDs  = nan(nWin, 1);
    termWs  = nan(nWin, 1);
    succ    = nan(nWin, 1);
    turns   = nan(nWin, 1);
    pooledNetRet = [];
    for ep = 1:nWin
        g = gross(ep, :)'; tu = turnover(ep, :)';
        if any(isnan(g)); continue; end
        netRet = g - c * tu;                             % net per-step simple return
        wealth = initialWealth * [1; cumprod(1 + netRet)];
        % Sharpe on log returns, matching diff(log(wealth)) in the training scripts
        logNet = diff(log(wealth));
        sharpes(ep) = mean(logNet) / std(logNet) * annualizeFactor;
        maxDDs(ep)  = max(cummax(wealth) - wealth) / max(wealth);
        termWs(ep)  = wealth(end);
        succ(ep)    = double(wealth(end) >= goalWealth);
        turns(ep)   = mean(tu);
        pooledNetRet = [pooledNetRet; logNet]; %#ok<AGROW>
    end
    meanSharpe = mean(sharpes, 'omitnan');
    p90MaxDD   = prctile(maxDDs, 90);
    p10TermW   = prctile(termWs, 10);
    succRate   = sum(succ, 'omitnan');                   % count out of nWin
    meanTurn   = mean(turns, 'omitnan');
end

function w = alphaToWeights(alpha, W_basis)
    % Mirror of week8_ddpg_onfrontier.m: linear interp over the (dense) basis.
    alpha = max(0, min(1, alpha));
    nCol  = size(W_basis, 2);
    idxF  = 1 + alpha * (nCol - 1);
    iLo   = max(1, floor(idxF));
    iHi   = min(nCol, iLo + 1);
    frac  = idxF - iLo;
    w     = (1 - frac) * W_basis(:, iLo) + frac * W_basis(:, iHi);
end
