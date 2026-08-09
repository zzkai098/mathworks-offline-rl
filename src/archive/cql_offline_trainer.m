function results = cql_offline_trainer(cqlAlpha, opts)
%CQL_OFFLINE_TRAINER  Offline CQL on top of the Week 8 on-frontier DDPG.
%
%   results = cql_offline_trainer(cqlAlpha)
%   results = cql_offline_trainer(cqlAlpha, opts)
%
% Motivation: vanilla DDPG has no offline-RL pessimism, so its actor slides
% to the out-of-distribution (OOD) Q-peak. In Week 8 this showed up as SEED
% COLLAPSE — 3 of 5 seeds saturated at alpha ~= 0 (full-defensive boundary).
% CQL adds a conservative critic penalty that pushes Q DOWN on sampled
% (random/OOD) actions and UP on the data action.
%
% `trainFromData` does NOT expose the critic loss, so CQL cannot be bolted
% onto it. This trainer replaces `trainFromData` with a hand-written
% dlfeval + adamupdate offline loop over raw dlnetworks. Everything ELSE is
% identical to week8_ddpg_onfrontier.m (data pipeline, dense on-frontier
% alpha map, regime-gated drawdown reward, 6D state, 5 seeds, eval) so this
% is a clean A/B whose ONLY new variable is cqlAlpha.
%
% Continuous CQL regularizer (CQL(H)-style, uniform action sampling):
%   critic loss = mean( (Q(s,a) - y)^2 )                     % standard TD
%               + cqlAlpha * mean( logmeanexp_{a'~U[0,1]} Q(s,a') - Q(s,a) )
%   actor loss  = -mean Q(s, mu(s))                          % standard DDPG
% with y = r + gamma*(1-done)*Q_targ(s', mu_targ(s'))  using BOTH the target
% actor and target critic (so cqlAlpha=0 is exactly DDPG — the anchor).
%
% NOTE (reviewer NICE-7): the uniform action samples carry NO importance-
% sampling correction. That is correct here ONLY because the action space is
% [0,1] with unit Lebesgue measure, so log q(a') = log 1 = 0. It would be
% WRONG for [0,1]^k or any non-unit-measure action scale.
%
% REVIEWER FIXES baked in vs the first draft:
%   - per-learnable L2 gradient clip = 1 on both nets (match week8 opts)
%   - dispersion = within-seed std of eval alpha across states (the real
%     hill-climb target; the quantized collapse rate is only REPORTED)
%   - Q(s,alpha~0) vs empirical discounted return-to-go diagnostic: the
%     BLOCKER-1 go/no-go that decides whether CQL is even the right tool
%
% opts fields (all optional):
%   .seeds        (default [1000 2000 3000 4000 5000])
%   .quickMode    (default false) -> 1 seed, few epochs, fast wiring test
%   .maxEpochs    (default 100)   FROZEN for the whole sweep
%   .stepsPerEpoch(default 400)
%   .nRandCQL     (default 10)    FROZEN for the whole sweep
%   .gradClip     (default 1)
%   .logsRoot     (default experiments/logs/cql_ddpg/alpha_<cqlAlpha>)
%   .verbose      (default true)
%   .collapseThr  (default 0.05)  low-boundary threshold for the reported rate
%   .aggBoundary  (default 0.95)  high-boundary threshold (upper collapse)
%
% results: cross-seed arrays (same fields as week8) PLUS
%   .cqlAlpha, .collapseRate, .aggCollapseRate, .collapseThr
%   .dispersionBySeed, .meanDispersion   (PRIMARY hill-climb statistic)
%   .meanAlphaBySeed, .finalCriticLoss, .finalCQLReg
%   .qDataAlpha0, .empRTGAlpha0, .qGapAlpha0   (BLOCKER-1 go/no-go)

if nargin < 2, opts = struct(); end
def = struct('seeds',[1000 2000 3000 4000 5000],'quickMode',false, ...
    'maxEpochs',100,'stepsPerEpoch',400,'nRandUnif',5,'nRandPolicy',5, ...
    'cqlSigma',0.2,'gradClip',1, ...
    'logsRoot',"",'verbose',true,'collapseThr',0.05,'aggBoundary',0.95);
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(opts,fn{k}) || isempty(opts.(fn{k}))
        opts.(fn{k}) = def.(fn{k});
    end
end
if opts.quickMode
    opts.seeds         = opts.seeds(1);
    opts.maxEpochs     = min(opts.maxEpochs, 8);
    opts.stepsPerEpoch = min(opts.stepsPerEpoch, 100);
end
if opts.logsRoot == ""
    opts.logsRoot = fullfile("experiments","logs","cql_ddpg", ...
        sprintf("alpha_%g", cqlAlpha));
end

%% Fixed parameters (mirror week8_ddpg_onfrontier.m)
initialWealth   = 100000;
goalWealth      = 102000;
contribution    = 0;

trainingRange   = 3020;
horizonPeriods  = 30;
numEpisodes     = 200;
numActions      = 15;
nDense          = 200;

hiddenUnits     = 32;
miniBatchSize   = 256;
discountFactor  = 0.995;
actorLR         = 1e-4;
criticLR        = 5e-4;
targetSmooth    = 5e-3;

seeds           = opts.seeds;
nSeeds          = numel(seeds);
numEvalEpisodes = 30;
evalStepSize    = 30;

BETA_HIGH = 8.0; BETA_LOW = 2.0; DD_THR = 0.03;
VIX_THR_Z = 1.5; SLOPE_THR_Z = -1.5;
COL_T10Y2Y = 2;  COL_VIX = 3;

%% Load data (identical to week8)
pricesTT   = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train    = pricesTT{:, 2:end};
R_train    = diff(log(P_train));
R_sub = R_train(1:trainingRange, :);
mu    = mean(R_sub, 1)';
Sigma = cov(R_sub);

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
M_train_z = (M_train - macroMean) ./ macroStd;

addpath(fullfile("src","utils"));
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', ...
                                    macroMean, macroStd);

isStressDay_train = (M_train_z(:, COL_VIX) > VIX_THR_Z) | ...
                    (M_train_z(:, COL_T10Y2Y) < SLOPE_THR_Z);
isStressDay_test  = (M_test_z(:, COL_VIX) > VIX_THR_Z) | ...
                    (M_test_z(:, COL_T10Y2Y) < SLOPE_THR_Z);

normalizeWealth = @(w) min(w/goalWealth, 5);
timeFrac        = @(t) t / horizonPeriods;
obsDim = 2 + size(M_train_z, 2);   % 6D
actDim = 1;

if opts.verbose
    fprintf("=== CQL offline trainer (CQL-on-DDPG) — cqlAlpha = %g ===\n", cqlAlpha);
    fprintf("  seeds=%s  quickMode=%d  maxEpochs=%d  stepsPerEpoch=%d  gradClip=%g\n", ...
        mat2str(seeds), opts.quickMode, opts.maxEpochs, opts.stepsPerEpoch, opts.gradClip);
    fprintf("  CQL-H estimator: %d uniform + %d policy(sigma=%.2f) samples w/ IS correction\n", ...
        opts.nRandUnif, opts.nRandPolicy, opts.cqlSigma);
    fprintf("  obsDim=%d actDim=%d  (cqlAlpha=0 => plain DDPG anchor)\n", obsDim, actDim);
end

%% Result storage
z = nan(nSeeds,1);
results = struct();
[results.successRate, results.meanSharpe, results.stdSharpe, results.meanMaxDD, ...
 results.stdMaxDD, results.meanTermWealth, results.p10TermWealth, results.p90TermWealth, ...
 results.minSharpe, results.p90MaxDD, results.meanAlphaAll, results.meanAlphaStress, ...
 results.meanAlphaNormal, results.finalCriticLoss, results.finalCQLReg, ...
 results.dispersionBySeed, results.qDataAlpha0, results.empRTGAlpha0] = deal(z);

%% Seed loop
for sIdx = 1:nSeeds
    seed = seeds(sIdx);
    rng(seed, "twister");
    if opts.verbose, fprintf("\n----- Seed %d (%d/%d) -----\n", seed, sIdx, nSeeds); end

    %% Build offline transition buffer (behavior policy: alpha ~ U[0,1])
    stepSize      = max(1, floor((trainingRange - horizonPeriods) / numEpisodes));
    sampledStarts = (1 + (0:numEpisodes-1)' * stepSize);
    sampledStarts = sampledStarts(sampledStarts + horizonPeriods - 1 <= trainingRange);
    nEp           = numel(sampledStarts);
    N             = nEp * horizonPeriods;

    bObs = zeros(obsDim, N); bAct = zeros(actDim, N); bRew = zeros(1, N);
    bNextObs = zeros(obsDim, N); bDone = zeros(1, N);
    idx = 0;
    for e = 1:nEp
        startIdx = sampledStarts(e);
        wealth = initialWealth; peakW = initialWealth;
        for t = 1:horizonPeriods
            dayIdx   = startIdx + t - 1;
            obs      = [normalizeWealth(wealth); timeFrac(t); M_train_z(dayIdx, :)'];
            alpha    = rand();
            w        = alphaToWeights(alpha, W_dense);
            r_tp1    = R_sub(dayIdx, :) * w;
            wealthNext = (wealth + contribution) * (1 + r_tp1);

            betaT   = BETA_HIGH * isStressDay_train(dayIdx) + BETA_LOW * ~isStressDay_train(dayIdx);
            peakW   = max(peakW, wealthNext);
            DD_t    = (peakW - wealthNext) / peakW;
            penalty = betaT * max(0, DD_t - DD_THR);
            reward  = log(wealthNext / wealth) - penalty;
            if t == horizonPeriods && wealthNext >= goalWealth
                reward = reward + 1.0;
            end

            nextDayIdx = min(dayIdx + 1, size(M_train_z, 1));
            nextObs = [normalizeWealth(wealthNext); timeFrac(min(t+1,horizonPeriods)); ...
                       M_train_z(nextDayIdx, :)'];

            idx = idx + 1;
            bObs(:,idx) = obs; bAct(:,idx) = alpha; bRew(idx) = reward;
            bNextObs(:,idx) = nextObs; bDone(idx) = (t == horizonPeriods);
            wealth = wealthNext;
        end
    end

    %% Build raw dlnetworks (same architecture as week8's actor/critic)
    actorNet = dlnetwork(layerGraph([
        featureInputLayer(obsDim, Name="state")
        fullyConnectedLayer(hiddenUnits)
        reluLayer
        fullyConnectedLayer(hiddenUnits)
        reluLayer
        fullyConnectedLayer(actDim)
        sigmoidLayer(Name="alpha")]));

    statePath = [featureInputLayer(obsDim, Name="state")
                 fullyConnectedLayer(hiddenUnits, Name="stateFC")];
    actionPath = [featureInputLayer(actDim, Name="action")
                  fullyConnectedLayer(hiddenUnits, Name="actionFC")];
    commonPath = [concatenationLayer(1, 2, Name="concat")
                  reluLayer
                  fullyConnectedLayer(hiddenUnits)
                  reluLayer
                  fullyConnectedLayer(1, Name="Qout")];
    criticLG = layerGraph(statePath);
    criticLG = addLayers(criticLG, actionPath);
    criticLG = addLayers(criticLG, commonPath);
    criticLG = connectLayers(criticLG, "stateFC", "concat/in1");
    criticLG = connectLayers(criticLG, "actionFC", "concat/in2");
    criticNet = dlnetwork(criticLG);

    actorTargetNet  = actorNet;
    criticTargetNet = criticNet;

    avgGA = []; avgSqGA = [];   % actor Adam state
    avgGC = []; avgSqGC = [];   % critic Adam state
    stepCount = 0;

    lastCriticLoss = NaN; lastCQLReg = NaN;
    totalSteps = opts.maxEpochs * opts.stepsPerEpoch;
    for gstep = 1:totalSteps
        mb = randi(N, [1, miniBatchSize]);
        dObs     = dlarray(bObs(:,mb), 'CB');
        dAct     = dlarray(bAct(:,mb), 'CB');
        dRew     = dlarray(bRew(mb), 'CB');
        dNextObs = dlarray(bNextObs(:,mb), 'CB');
        dDone    = dlarray(bDone(mb), 'CB');

        % --- Critic update (TD + CQL-H) ---
        [gC, tdLoss, cqlReg] = dlfeval(@criticGradients, criticNet, actorNet, ...
            actorTargetNet, criticTargetNet, dObs, dAct, dRew, dNextObs, dDone, ...
            discountFactor, cqlAlpha, opts.nRandUnif, opts.nRandPolicy, opts.cqlSigma);
        gC = dlupdate(@(g) clipL2(g, opts.gradClip), gC);      % per-learnable L2 clip
        stepCount = stepCount + 1;
        [criticNet, avgGC, avgSqGC] = adamupdate(criticNet, gC, avgGC, avgSqGC, ...
            stepCount, criticLR);

        % --- Actor update ---
        gA = dlfeval(@actorGradients, actorNet, criticNet, dObs);
        gA = dlupdate(@(g) clipL2(g, opts.gradClip), gA);
        [actorNet, avgGA, avgSqGA] = adamupdate(actorNet, gA, avgGA, avgSqGA, ...
            stepCount, actorLR);

        % --- Soft target update (both nets) ---
        criticTargetNet = softUpdate(criticTargetNet, criticNet, targetSmooth);
        actorTargetNet  = softUpdate(actorTargetNet,  actorNet,  targetSmooth);

        lastCriticLoss = double(gather(extractdata(tdLoss)));
        lastCQLReg     = double(gather(extractdata(cqlReg)));
        if opts.verbose && mod(gstep, max(1,round(totalSteps/5))) == 0
            fprintf("    gstep %5d/%d  tdLoss=%.4e  cqlReg=%+.4e\n", ...
                gstep, totalSteps, lastCriticLoss, lastCQLReg);
        end
    end
    results.finalCriticLoss(sIdx) = lastCriticLoss;
    results.finalCQLReg(sIdx)     = lastCQLReg;

    %% BLOCKER-1 go/no-go diagnostic: Q(s, alpha~0) vs empirical return-to-go
    % Reshape buffer into [horizon x nEp] to get discounted return-to-go per
    % transition, then restrict to near-alpha=0 data actions and compare the
    % TRAINED critic's Q against the realized (behavior-policy) return-to-go.
    %   Q >> emp  -> critic over-extrapolates the defensive corner -> CQL apt
    %   Q ~= emp  -> boundary is data-justified -> CQL is the wrong tool
    Rmat = reshape(bRew, horizonPeriods, nEp);
    RTG  = zeros(size(Rmat));
    running = zeros(1, nEp);
    for t = horizonPeriods:-1:1
        running = Rmat(t,:) + discountFactor * running;
        RTG(t,:) = running;
    end
    RTGflat = reshape(RTG, 1, N);
    near0   = bAct < 0.10;
    if any(near0)
        obs0 = dlarray(bObs(:, near0), 'CB');
        act0 = dlarray(bAct(:, near0), 'CB');
        q0   = predict(criticNet, obs0, act0);
        results.qDataAlpha0(sIdx)  = double(mean(gather(extractdata(q0))));
        results.empRTGAlpha0(sIdx) = mean(RTGflat(near0));
    end

    %% Save trained agent (wrap raw nets into rlDDPGAgent for harness reuse)
    seedLogsFolder = fullfile(opts.logsRoot, sprintf("seed_%d", seed));
    if ~isfolder(seedLogsFolder), mkdir(seedLogsFolder); end
    obsInfo = rlNumericSpec([obsDim 1], Name="GBWMObservation");
    actInfo = rlNumericSpec([actDim 1], Name="RiskPosition", LowerLimit=0, UpperLimit=1);
    actor  = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
                ObservationInputNames="state");
    critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
                ObservationInputNames="state", ActionInputNames="action");
    aopts = rlDDPGAgentOptions(DiscountFactor=discountFactor, ...
        MiniBatchSize=miniBatchSize, TargetSmoothFactor=targetSmooth);
    agent = rlDDPGAgent(actor, critic, aopts);
    save(fullfile(seedLogsFolder, "TrainedAgent.mat"), "agent");

    %% Hold-out eval (identical protocol to week8)
    successCount = 0;
    allSharpes = nan(numEvalEpisodes,1); allMaxDD = nan(numEvalEpisodes,1);
    allTermWealth = nan(numEvalEpisodes,1);
    allAlphas = nan(numEvalEpisodes, horizonPeriods);
    allStress = false(numEvalEpisodes, horizonPeriods);
    for ep = 1:numEvalEpisodes
        startEval = (ep-1)*evalStepSize + 1;
        if startEval + horizonPeriods - 1 > size(R_full, 1), break; end
        wealth = zeros(horizonPeriods+1,1); wealth(1) = initialWealth;
        for t = 1:horizonPeriods
            dayIdx = startEval + t - 1;
            obs    = dlarray([normalizeWealth(wealth(t)); timeFrac(t); M_test_z(dayIdx,:)'], 'CB');
            alpha  = double(gather(extractdata(predict(actorNet, obs))));
            alpha  = max(0, min(1, alpha));
            allAlphas(ep,t) = alpha; allStress(ep,t) = isStressDay_test(dayIdx);
            w      = alphaToWeights(alpha, W_dense);
            wealth(t+1) = (wealth(t) + contribution) * (1 + R_full(dayIdx,:) * w);
        end
        rets   = diff(log(wealth));
        allSharpes(ep)    = mean(rets)/std(rets)*sqrt(252);
        allMaxDD(ep)      = max(cummax(wealth) - wealth) / max(wealth);
        allTermWealth(ep) = wealth(end);
        successCount = successCount + double(wealth(end) >= goalWealth);
    end

    results.successRate(sIdx)    = successCount;
    results.meanSharpe(sIdx)     = mean(allSharpes,'omitnan');
    results.stdSharpe(sIdx)      = std(allSharpes,'omitnan');
    results.meanMaxDD(sIdx)      = mean(allMaxDD,'omitnan');
    results.stdMaxDD(sIdx)       = std(allMaxDD,'omitnan');
    results.meanTermWealth(sIdx) = mean(allTermWealth,'omitnan');
    results.p10TermWealth(sIdx)  = prctile(allTermWealth,10);
    results.p90TermWealth(sIdx)  = prctile(allTermWealth,90);
    results.minSharpe(sIdx)      = min(allSharpes);
    results.p90MaxDD(sIdx)       = prctile(allMaxDD,90);
    fa = allAlphas(:); fs = allStress(:);
    results.meanAlphaAll(sIdx)    = mean(fa,'omitnan');
    results.meanAlphaStress(sIdx) = mean(fa(fs),'omitnan');
    results.meanAlphaNormal(sIdx) = mean(fa(~fs),'omitnan');
    % PRIMARY hill-climb statistic: within-seed dispersion of eval alpha
    % across states. Collapsed (either boundary or flat-middle) -> ~0.
    results.dispersionBySeed(sIdx) = std(fa,'omitnan');

    if opts.verbose
        fprintf("  Success %d/%d | Sharpe %.2f | MaxDD %.2f%% | P90DD %.2f%% | TermW P10 %.0f\n", ...
            successCount, numEvalEpisodes, results.meanSharpe(sIdx), ...
            results.meanMaxDD(sIdx)*100, results.p90MaxDD(sIdx)*100, results.p10TermWealth(sIdx));
        fprintf("  meanAlpha %.3f (S %.3f/N %.3f) | dispersion %.3f | Q(a~0) %.3f vs empRTG %.3f\n", ...
            results.meanAlphaAll(sIdx), results.meanAlphaStress(sIdx), ...
            results.meanAlphaNormal(sIdx), results.dispersionBySeed(sIdx), ...
            results.qDataAlpha0(sIdx), results.empRTGAlpha0(sIdx));
    end
end

%% Cross-seed aggregate
results.cqlAlpha         = cqlAlpha;
results.collapseThr      = opts.collapseThr;
results.meanAlphaBySeed  = results.meanAlphaAll;
results.collapseRate     = mean(results.meanAlphaAll < opts.collapseThr);       % low-boundary
results.aggCollapseRate  = mean(results.meanAlphaAll > opts.aggBoundary);       % high-boundary
results.meanDispersion   = mean(results.dispersionBySeed,'omitnan');            % PRIMARY
results.qGapAlpha0        = mean(results.qDataAlpha0 - results.empRTGAlpha0,'omitnan');

if opts.verbose
    fprintf("\n===== CQL cqlAlpha=%g cross-seed (n=%d) =====\n", cqlAlpha, nSeeds);
    fprintf("  PRIMARY meanDispersion = %.3f  (higher = state-responsive; ~0 = collapsed)\n", ...
        results.meanDispersion);
    fprintf("  Reported collapse: low %.0f%% (meanAlpha<%.2f) / high %.0f%% (meanAlpha>%.2f)  [week8 low: 60%%]\n", ...
        100*results.collapseRate, opts.collapseThr, 100*results.aggCollapseRate, opts.aggBoundary);
    fprintf("  Success %.1f/%d | MeanSharpe %.2f (std %.2f) | MaxDD P90 %.2f%% | TermW P10 %.0f\n", ...
        mean(results.successRate), numEvalEpisodes, mean(results.meanSharpe), ...
        std(results.meanSharpe), mean(results.p90MaxDD)*100, mean(results.p10TermWealth));
    fprintf("  Worst Sharpe %.2f | Alpha S/N %.3f/%.3f (diff %+.3f)\n", ...
        mean(results.minSharpe), mean(results.meanAlphaStress,'omitnan'), ...
        mean(results.meanAlphaNormal,'omitnan'), ...
        mean(results.meanAlphaStress,'omitnan')-mean(results.meanAlphaNormal,'omitnan'));
    fprintf("  >>> GO/NO-GO  Q(a~0)=%.3f  empRTG=%.3f  gap=%+.3f  (gap>>0 => CQL apt; gap~0 => data-justified, STOP)\n", ...
        mean(results.qDataAlpha0,'omitnan'), mean(results.empRTGAlpha0,'omitnan'), results.qGapAlpha0);
end
end


%% ===== Local functions =====
function [grad, tdLoss, cqlReg] = criticGradients(criticNet, actorNet, ...
        actorTargetNet, criticTargetNet, obs, act, rew, nextObs, done, ...
        gamma, cqlAlpha, nU, nP, sigma)
    % TD target using BOTH target actor and target critic (=> cqlAlpha=0 is DDPG)
    % Target nets are separate dlnetworks; their learnables are NOT the
    % differentiation variable, so y carries no gradient w.r.t. criticNet —
    % no explicit stop-gradient needed.
    nextAct = predict(actorTargetNet, nextObs);
    qNext   = predict(criticTargetNet, nextObs, nextAct);
    y       = rew + gamma .* (1 - done) .* qNext;

    qData   = forward(criticNet, obs, act);
    tdLoss  = mean((qData - y).^2, 'all');

    % Continuous CQL(H): estimate log∫exp Q(s,a) da by importance sampling
    % from a MIXTURE of proposals, so the penalty specifically targets the
    % actor's own action region (the uniform-only variant was action-nearly-
    % uniform and merely relocated the collapse — reviewer NICE-7).
    %   term_j = Q(s, a_j) - log q(a_j) ;  penalty = mean(logmeanexp_j term_j - Q_data)
    B = size(obs, 2);
    terms = cell(nU + nP, 1);

    % (1) Uniform proposal on [0,1]: log q = 0 (unit Lebesgue measure).
    for j = 1:nU
        aU = dlarray(rand(1, B, 'like', extractdata(obs)), 'CB');
        terms{j} = forward(criticNet, obs, aU);
    end
    % (2) Current-policy proposal mu(s)+sigma*N(0,1), Gaussian IS correction.
    % predict() => mu treated as a constant w.r.t. the critic gradient.
    muS     = predict(actorNet, obs);
    logNorm = log(sigma) + 0.5*log(2*pi);
    for j = 1:nP
        eps  = dlarray(randn(1, B, 'like', extractdata(obs)), 'CB');
        aP   = max(0, min(1, muS + sigma*eps));
        logq = -0.5*(eps.^2) - logNorm;                 % log N(aP; mu, sigma^2)
        terms{nU+j} = forward(criticNet, obs, aP) - logq;
    end

    stack = cat(1, terms{:});                            % [(nU+nP) x B]
    m     = max(stack, [], 1);
    lse   = m + log(sum(exp(stack - m), 1));
    logIntExp = lse - log(nU + nP);
    cqlReg    = mean(logIntExp - qData, 'all');

    criticLoss = tdLoss + cqlAlpha * cqlReg;
    grad = dlgradient(criticLoss, criticNet.Learnables);
end

function grad = actorGradients(actorNet, criticNet, obs)
    piAct = forward(actorNet, obs);
    qPi   = forward(criticNet, obs, piAct);
    actorLoss = -mean(qPi, 'all');
    grad = dlgradient(actorLoss, actorNet.Learnables);
end

function net = softUpdate(net, srcNet, tau)
    net.Learnables = dlupdate(@(t,s) (1-tau)*t + tau*s, ...
        net.Learnables, srcNet.Learnables);
end

function g = clipL2(g, thr)
    % Per-learnable L2-norm gradient clipping (matches week8's
    % GradientThreshold=1 with the default 'l2norm' method).
    n = sqrt(sum(g.^2, 'all'));
    if extractdata(n) > thr
        g = g * (thr / n);
    end
end

function w = alphaToWeights(alpha, W_frontier)
    alpha = max(0, min(1, alpha));
    nCol  = size(W_frontier, 2);
    idxF  = 1 + alpha * (nCol - 1);
    iLo   = max(1, floor(idxF));
    iHi   = min(nCol, iLo + 1);
    frac  = idxF - iLo;
    w     = (1 - frac) * W_frontier(:, iLo) + frac * W_frontier(:, iHi);
end
