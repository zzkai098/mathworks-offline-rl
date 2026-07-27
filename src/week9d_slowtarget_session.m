%% Week 9d — Convergence recipe LEVER #1: slow the target update
% Identical to week9c (which is 6.1-at-ME-epochs, confound-free one-process-per-run)
% EXCEPT one variable: the target network update.
%   week9c / 6.1 : TargetUpdateMethod="periodic", TargetUpdateFrequency=4
%                  (hard-copy the target net every 4 gradient steps — the target
%                   barely lags the online net, so it fails to anchor the bootstrap;
%                   week9c showed Q diverging to 1e3-1e5 across all seeds).
%   week9d       : TargetUpdateMethod="smoothing", TargetSmoothFactor=TSF (=1e-3)
%                  (Polyak averaging — the target moves ~1000x slower, restoring the
%                   stable anchor that should bound the Bellman recursion).
%
% Hypothesis (from subagent review of the week9c divergence): the 1e5 blow-up is an
% UNSTABLE recursion (bootstrap gain>1) driven mainly by the over-fast target update,
% not by OOD over-optimism alone. If slowing the target bounds Q, vanilla DQN
% converges and the 6.1 DQN lineage is preserved (IQL/CQL become improvements, not
% rescues). ONE lever changed so the effect is cleanly attributable.
%
% Usage:  SEED=<s>; ME=<e>; TSF=<tau>; week9d_slowtarget_session   (defaults 1000/100/1e-3)
% Appends one row per run to experiments/logs/week9d_results.csv (header from driver).

if ~exist('ME', 'var');   ME = 100;    end
if ~exist('SEED', 'var'); SEED = 1000; end
if ~exist('TSF', 'var');  TSF = 1e-3;  end
seed = SEED;

%% Fixed Parameters (identical to week6_1_regime_gated.m / week9c)
initialWealth = 100000;  goalWealth = 102000;  contribution = 0;
trainingRange = 3020;    horizonPeriods = 30;  numEpisodes = 200;  numActions = 15;
hiddenUnits = 32;        miniBatchSize = 256;  discountFactor = 0.995;
stepsPerEpoch = 400;
BETA_HIGH = 8.0;  BETA_LOW = 2.0;  DD_THR = 0.03;
VIX_THR_Z = 1.5;  SLOPE_THR_Z = -1.5;  COL_T10Y2Y = 2;  COL_VIX = 3;
N_VAL = 300;

logsRoot   = fullfile("experiments", "logs", "week9d", sprintf("seed_%d_ME_%d_tsf_%g", seed, ME, TSF));
resultsCsv = fullfile("experiments", "logs", "week9d_results.csv");
numEvalEpisodes = 30;  evalStepSize = 30;

%% Train prices + frontier
pricesTT   = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train = pricesTT{:, 2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange, :);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
pf = Portfolio("AssetList", assetNames);
pf = setDefaultConstraints(pf);  pf = setAssetMoments(pf, mu, Sigma);
W_frontier = estimateFrontier(pf, numActions);

%% Macro + stress
macroTT = readtable("data/macro_train_extended.csv");
M_train = macroTT{:, 2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
M_train_z = (M_train - macroMean) ./ macroStd;
isStressDay_train = (M_train_z(:,COL_VIX) > VIX_THR_Z) | (M_train_z(:,COL_T10Y2Y) < SLOPE_THR_Z);

normalizeWealth = @(w) min(w/goalWealth, 5);
timeFrac        = @(t) t / horizonPeriods;
obsDim          = 2 + size(M_train_z, 2);

%% Generate + freeze episodes (deterministic, rng(seed)) — identical to week9c
rng(seed, "twister");
if isfolder(logsRoot); delete(fullfile(logsRoot, "*.mat")); else; mkdir(logsRoot); end
stepSize = max(1, floor((trainingRange - horizonPeriods) / numEpisodes));
starts   = (1 + (0:numEpisodes-1)' * stepSize);
starts   = starts(starts + horizonPeriods - 1 <= trainingRange);
nEp      = numel(starts);

valObsAll = zeros(obsDim, nEp*horizonPeriods);  valTimeIdx = zeros(1, nEp*horizonPeriods);
cursor = 0;
for ec = 1:nEp
    startIdx = starts(ec);
    expStruct = repmat(struct("Observation",[],"Action",[],"Reward",[], ...
        "NextObservation",[],"IsDone",[]), horizonPeriods, 1);
    wealth = initialWealth;  peakW = initialWealth;
    for t = 1:horizonPeriods
        dayIdx = startIdx + t - 1;
        obs = [normalizeWealth(wealth); timeFrac(t); M_train_z(dayIdx,:)'];
        aIdx = randi(numActions);  w = W_frontier(:, aIdx);
        wealthNext = (wealth + contribution) * (1 + R_sub(dayIdx,:) * w);
        stressFlag = isStressDay_train(dayIdx);
        betaT = BETA_HIGH*stressFlag + BETA_LOW*~stressFlag;
        peakW = max(peakW, wealthNext);  DD_t = (peakW - wealthNext)/peakW;
        reward = log(wealthNext/wealth) - betaT*max(0, DD_t - DD_THR);
        if t == horizonPeriods && wealthNext >= goalWealth; reward = reward + 1.0; end
        nextDayIdx = min(dayIdx+1, size(M_train_z,1));
        nextObs = [normalizeWealth(wealthNext); timeFrac(min(t+1,horizonPeriods)); M_train_z(nextDayIdx,:)'];
        expStruct(t).Observation={obs}; expStruct(t).Action={aIdx};
        expStruct(t).Reward=reward; expStruct(t).NextObservation={nextObs};
        expStruct(t).IsDone=(t==horizonPeriods);
        cursor = cursor+1; valObsAll(:,cursor)=obs; valTimeIdx(cursor)=t;
        wealth = wealthNext;
    end
    exp = expStruct;
    save(fullfile(logsRoot, sprintf("loggedData%03d.mat", ec)), "exp");
end
nSteps = cursor;  valObsAll = valObsAll(:,1:nSteps);  valTimeIdx = valTimeIdx(1:nSteps);

%% Frozen val set (train-only, deterministic)
rng(seed + 777, "twister");
valSel = randperm(nSteps, min(N_VAL, nSteps));
valObs = valObsAll(:, valSel);  valT = valTimeIdx(valSel);
isEarly = valT <= 10;  isLate = valT >= 20;

%% Build agent — ONLY CHANGE vs week9c: Polyak (smoothing) target update
rng(seed, "twister");
myReadFcn = @(fname) load(fname, "exp").exp;
fds = fileDatastore(fullfile(logsRoot, "loggedData*.mat"), "ReadFcn", myReadFcn);
reset(fds);
obsInfo = rlNumericSpec([obsDim 1], Name="GBWMObservation");
actInfo = rlFiniteSetSpec(1:numActions, Name="AllocationIndex");
criticLG = layerGraph([
    featureInputLayer(obsDim, Name="state")
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(numActions, Name="Qout")]);
critic = rlVectorQValueFunction(dlnetwork(criticLG), obsInfo, actInfo);
agentOpts = rlDQNAgentOptions(DiscountFactor=discountFactor, ExperienceBufferLength=1e6, ...
    MiniBatchSize=miniBatchSize, TargetUpdateMethod="smoothing", TargetSmoothFactor=TSF, ...
    UseDoubleDQN=true);
agentOpts.CriticOptimizerOptions.LearnRate = 5e-4;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1;
agent = rlDQNAgent(critic, agentOpts);

%% SINGLE trainFromData call (like 6.1, ME epochs)
fprintf("=== week9d: seed %d, ME %d, TargetSmoothFactor %g (single call) ===\n", seed, ME, TSF);
tfdOpts = rlTrainingFromDataOptions(MaxEpochs=ME, NumStepsPerEpoch=stepsPerEpoch, ...
    Plots="none", Verbose=false);
rng(seed + 12345, "twister");
trainFromData(agent, fds, tfdOpts);

%% Train-only val-Q + discrimination
critic = getCritic(agent);
qAll = getValue(critic, {valObs});
assert(isequal(size(qAll), [numActions, numel(valSel)]), "getValue shape unexpected");
maxQ = max(qAll, [], 1);
qG = mean(maxQ);  qE = mean(maxQ(isEarly));  qL = mean(maxQ(isLate));
sortQ = sort(qAll, 1, "descend");
margin = mean(sortQ(1,:) - sortQ(2,:));
stateStd = std(maxQ);
marginN  = margin / max(stateStd, eps);
[~, gA] = max(qAll, [], 1);
effA = numel(unique(gA));

%% Hold-out eval (description only)
[R_full, M_test_z] = localTestData('data/prices_test.csv', 'data/macro_test.csv', macroMean, macroStd);
succ = 0; sharpes = nan(numEvalEpisodes,1); mdd = nan(numEvalEpisodes,1); termW = nan(numEvalEpisodes,1);
for ep = 1:numEvalEpisodes
    s0 = (ep-1)*evalStepSize + 1;
    if s0 + horizonPeriods - 1 > size(R_full,1); break; end
    wpath = zeros(horizonPeriods+1,1); wpath(1) = initialWealth;
    for t = 1:horizonPeriods
        dIdx = s0 + t - 1;
        obs = { [normalizeWealth(wpath(t)); timeFrac(t); M_test_z(dIdx,:)'] };
        a = agent.getAction(obs);
        wpath(t+1) = (wpath(t) + contribution) * (1 + R_full(dIdx,:) * W_frontier(:, a{1}));
    end
    rets = diff(log(wpath));
    sharpes(ep) = mean(rets)/std(rets)*sqrt(252);
    mdd(ep) = max(cummax(wpath)-wpath)/max(wpath);
    termW(ep) = wpath(end);
    succ = succ + double(wpath(end) >= goalWealth);
end
evalSharpe = mean(sharpes,'omitnan');
evalMddP90 = prctile(mdd, 90);
evalTermP10 = prctile(termW, 10);

%% Report + append CSV (tsf as first column for A/B vs week9c)
fprintf("  val: Q %.4f (early %.4f / late %.4f) | margin %.4f | stateStd %.4f | margin/stateStd %.4f | effA %d/%d\n", ...
    qG, qE, qL, margin, stateStd, marginN, effA, numActions);
fprintf("  eval(descr): Sharpe %.3f | success %d/%d | MaxDD P90 %.2f%% | TermW P10 %.0f\n", ...
    evalSharpe, succ, numEvalEpisodes, 100*evalMddP90, evalTermP10);

row = [TSF, seed, ME, qG, qE, qL, margin, stateStd, marginN, effA, evalSharpe, succ, evalMddP90, evalTermP10];
writematrix(row, resultsCsv, "WriteMode", "append");
fprintf("  appended -> %s\n", resultsCsv);


%% local test-data loader (same as 6.1)
function [R, M_z] = localTestData(priceFile, macroFile, macroMean, macroStd)
    pricesTT = readtable(priceFile, 'VariableNamingRule', 'preserve');
    R = diff(log(pricesTT{:, 2:end}));
    macroTT = readtable(macroFile, 'VariableNamingRule', 'preserve');
    M_z = (macroTT{:, 2:end} - macroMean) ./ macroStd;
end
