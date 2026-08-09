%% Week 10 — IQL (Implicit Q-Learning) on the 6.1 GBWM setup
% The deliverable learner. Diagnosis (week9c/d/f) proved 6.1's DQN diverges via the
% OOD max-bootstrap; slowing the target (Polyak) only delays it. IQL removes that term
% STRUCTURALLY: it regresses an expectile of V(s) from IN-DATASET actions and bootstraps
% Q from V(s') — the exploding max_a Q(s',OOD) never appears.
%
% Same everything as 6.1/week9d: extended train 2010-2021, 6D state, 15 frontier actions,
% regime-gated drawdown reward + terminal bonus, gamma=0.995, Polyak target tau=1e-3.
% ONLY the value-learning rule changes (custom dlfeval loop, not trainFromData).
%
% Two losses (Kostrikov et al. 2021):
%   V-loss (expectile regression):  L_V = mean( w .* (Qtarget(s,a) - V(s))^2 ),
%                                   w = |EXPECTILE - 1(u<0)|,  u = Qtarget(s,a)-V(s)
%                                   -> V approaches an upper expectile of Q at dataset actions.
%   Q-loss (TD bootstrapping V):    L_Q = mean( (Q(s,a) - [r + gamma*(1-done)*V(s')])^2 )
% Q and V updated by SEPARATE dlfeval calls -> the Q-gradient is taken w.r.t. qNet only,
% so V(s') is effectively a constant target (no detach hack needed). Polyak target on Q.
% Greedy policy = argmax_a Q(s,a) (in discrete IQL, argmax Q == argmax advantage since V(s)
% is action-independent), so AWR temperature beta is not needed for greedy eval.
%
% Confound-free harness kept: one fresh MATLAB process per (SEED, ME); controlled
% minibatch RNG for reproducibility. Run via a driver like run_week9c.sh.
%
% Usage: SEED=<s>; ME=<e>; EXPECTILE=<0.7>; TSF=<1e-3>; LRATE=<3e-4>; OUTCSV=<path>; week10_iql_session

if ~exist('ME','var');        ME = 100;      end
if ~exist('SEED','var');      SEED = 1000;   end
if ~exist('EXPECTILE','var'); EXPECTILE = 0.7; end
if ~exist('TSF','var');       TSF = 1e-3;    end   % Polyak target-Q smoothing
if ~exist('LRATE','var');     LRATE = 3e-4;  end
seed = SEED;

%% Fixed Parameters (identical to 6.1 / week9d)
initialWealth = 100000;  goalWealth = 102000;  contribution = 0;
trainingRange = 3020;    horizonPeriods = 30;  numEpisodes = 200;  numActions = 15;
hiddenUnits = 32;        miniBatchSize = 256;  discountFactor = 0.995;
stepsPerEpoch = 400;     % => numUpdates = ME*stepsPerEpoch (match DQN budget)
BETA_HIGH = 8.0;  BETA_LOW = 2.0;  DD_THR = 0.03;
VIX_THR_Z = 1.5;  SLOPE_THR_Z = -1.5;  COL_T10Y2Y = 2;  COL_VIX = 3;
N_VAL = 300;

if exist('OUTCSV','var'); resultsCsv = OUTCSV;
else; resultsCsv = fullfile("experiments","logs","week10_iql_results.csv"); end
numEvalEpisodes = 30;  evalStepSize = 30;

%% Train prices + frontier
pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
pf = Portfolio("AssetList", assetNames);
pf = setDefaultConstraints(pf);  pf = setAssetMoments(pf, mu, Sigma);
W_frontier = estimateFrontier(pf, numActions);

%% Macro + stress
macroTT = readtable("data/macro_train_extended.csv");
M_train = macroTT{:,2:end};
macroMean = mean(M_train,1); macroStd = std(M_train,0,1); macroStd(macroStd==0)=1;
M_train_z = (M_train - macroMean) ./ macroStd;
isStressDay_train = (M_train_z(:,COL_VIX) > VIX_THR_Z) | (M_train_z(:,COL_T10Y2Y) < SLOPE_THR_Z);

normalizeWealth = @(w) min(w/goalWealth, 5);
timeFrac = @(t) t / horizonPeriods;
obsDim = 2 + size(M_train_z,2);

%% Generate episodes directly into transition ARRAYS (custom loop, no fileDatastore)
rng(seed, "twister");
stepSize = max(1, floor((trainingRange - horizonPeriods)/numEpisodes));
starts = (1 + (0:numEpisodes-1)'*stepSize);
starts = starts(starts + horizonPeriods - 1 <= trainingRange);
nEp = numel(starts);
N = nEp*horizonPeriods;

S    = zeros(obsDim, N);   % state
Aidx = zeros(1, N);        % action index (1..numActions)
Rwd  = zeros(1, N);        % reward
SP   = zeros(obsDim, N);   % next state
Dn   = zeros(1, N);        % done flag
Tt   = zeros(1, N);        % step index (for val buckets)
logRetAll = zeros(1, N);
k = 0;
for ec = 1:nEp
    startIdx = starts(ec);  wealth = initialWealth;  peakW = initialWealth;
    for t = 1:horizonPeriods
        dayIdx = startIdx + t - 1;
        obs = [normalizeWealth(wealth); timeFrac(t); M_train_z(dayIdx,:)'];
        a = randi(numActions);  w = W_frontier(:,a);
        wealthNext = (wealth + contribution)*(1 + R_sub(dayIdx,:)*w);
        logRet = log(wealthNext/wealth);
        stressFlag = isStressDay_train(dayIdx);
        betaT = BETA_HIGH*stressFlag + BETA_LOW*~stressFlag;
        peakW = max(peakW, wealthNext);  DD_t = (peakW - wealthNext)/peakW;
        reward = logRet - betaT*max(0, DD_t - DD_THR);
        if t==horizonPeriods && wealthNext>=goalWealth; reward = reward + 1.0; end
        nd = min(dayIdx+1, size(M_train_z,1));
        nextObs = [normalizeWealth(wealthNext); timeFrac(min(t+1,horizonPeriods)); M_train_z(nd,:)'];
        k = k+1;
        S(:,k)=obs; Aidx(k)=a; Rwd(k)=reward; SP(:,k)=nextObs; Dn(k)=(t==horizonPeriods);
        Tt(k)=t; logRetAll(k)=logRet;
        wealth = wealthNext;
    end
end
% one-hot action selector (constant) for gathering Q(s,a)
Aoh = zeros(numActions, N);  Aoh(sub2ind([numActions N], Aidx, 1:N)) = 1;

%% Frozen val set (train-only, deterministic)
rng(seed+777, "twister");
valSel = randperm(N, min(N_VAL,N));
valS = dlarray(S(:,valSel), 'CB');
valT = Tt(valSel);  isEarly = valT<=10;  isLate = valT>=20;
qCeiling = horizonPeriods*median(abs(logRetAll)) + 1.0;   % |Q| economic ceiling (bonus=1.0)

%% Build Q, V, target-Q networks (dlnetwork; custom loop)
rng(seed, "twister");
qNet = dlnetwork(layerGraph([
    featureInputLayer(obsDim, Name="s")
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(numActions, Name="Q")]));
vNet = dlnetwork(layerGraph([
    featureInputLayer(obsDim, Name="s")
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(1, Name="V")]));
qTgt = qNet;   % target Q (Polyak)

qAvgG=[]; qAvgSqG=[]; vAvgG=[]; vAvgSqG=[];

%% Custom IQL training loop
numUpdates = ME*stepsPerEpoch;
rng(seed+12345, "twister");   % controlled minibatch RNG for reproducibility
for it = 1:numUpdates
    mb = randi(N, [1 miniBatchSize]);   % ROW -> Rwd(mb)/Dn(mb) stay [1 x B]
    s   = dlarray(S(:,mb), 'CB');
    sp  = dlarray(SP(:,mb), 'CB');
    aoh = dlarray(Aoh(:,mb), 'CB');
    r   = dlarray(Rwd(mb), 'CB');       % [1 x B]
    dn  = dlarray(Dn(mb), 'CB');

    % --- V update (expectile regression of target-Q(s,a) toward V(s)) ---
    [lossV, gV] = dlfeval(@vGrad, vNet, qTgt, s, aoh, EXPECTILE);
    [vNet, vAvgG, vAvgSqG] = adamupdate(vNet, gV, vAvgG, vAvgSqG, it, LRATE);

    % --- Q update (TD bootstrapping V(s'); grad only w.r.t. qNet => V(s') is constant) ---
    [lossQ, gQ] = dlfeval(@qGrad, qNet, vNet, s, aoh, r, sp, dn, discountFactor);
    [qNet, qAvgG, qAvgSqG] = adamupdate(qNet, gQ, qAvgG, qAvgSqG, it, LRATE);

    % --- Polyak target-Q update:  theta_tgt <- (1-tau)*theta_tgt + tau*theta ---
    qTgt.Learnables = dlupdate(@(tp,p) (1-TSF).*tp + TSF.*p, qTgt.Learnables, qNet.Learnables);

    if it == 1
        % Guard: qTgt must be an INDEPENDENT copy (dlnetwork value semantics), else Polyak
        % is a no-op / target==online and the divergence we are fixing comes back.
        assert(~isequaln(extractdata(qTgt.Learnables.Value{1}), extractdata(qNet.Learnables.Value{1})), ...
            "qTgt aliased to qNet: target net not independent — Polyak update ineffective.");
    end
end

%% Train-only val-Q + discrimination
qValAll = extractdata(forward(qNet, valS));   % [numActions x N_VAL]
maxQ = max(qValAll, [], 1);
qG = mean(maxQ);  qE = mean(maxQ(isEarly));  qL = mean(maxQ(isLate));
vVal = extractdata(forward(vNet, valS));      % [1 x N_VAL]
vMean = mean(vVal);
sortQ = sort(qValAll, 1, 'descend');
margin = mean(sortQ(1,:) - sortQ(2,:));
stateStd = std(maxQ);
[~, gA] = max(qValAll, [], 1);
effA = numel(unique(gA));

%% Hold-out eval (greedy argmax Q) — description only
[R_full, M_test_z] = localTestData('data/prices_test.csv','data/macro_test.csv',macroMean,macroStd);
succ=0; sharpes=nan(numEvalEpisodes,1); mdd=nan(numEvalEpisodes,1); termW=nan(numEvalEpisodes,1);
for ep=1:numEvalEpisodes
    s0=(ep-1)*evalStepSize+1;
    if s0+horizonPeriods-1 > size(R_full,1); break; end
    wpath=zeros(horizonPeriods+1,1); wpath(1)=initialWealth;
    for t=1:horizonPeriods
        dIdx=s0+t-1;
        obs=[normalizeWealth(wpath(t)); timeFrac(t); M_test_z(dIdx,:)'];
        qv=extractdata(forward(qNet, dlarray(obs,'CB')));
        [~,a]=max(qv);
        wpath(t+1)=(wpath(t)+contribution)*(1 + R_full(dIdx,:)*W_frontier(:,a));
    end
    rets=diff(log(wpath));
    sharpes(ep)=mean(rets)/std(rets)*sqrt(252);
    mdd(ep)=max(cummax(wpath)-wpath)/max(wpath);
    termW(ep)=wpath(end);
    succ=succ+double(wpath(end)>=goalWealth);
end
evalSharpe=mean(sharpes,'omitnan'); evalMddP90=prctile(mdd,90); evalTermP10=prctile(termW,10);

%% Report + append CSV
fprintf("=== week10 IQL: seed %d ME %d expectile %.2f tau %g LR %g ===\n", seed, ME, EXPECTILE, TSF, LRATE);
fprintf("  val: Q %.4f (early %.4f/late %.4f) | V %.4f | margin %.4f | stateStd %.4f | effA %d/%d | qCeiling %.3f (Q=%.1fx)\n", ...
    qG, qE, qL, vMean, margin, stateStd, effA, numActions, qCeiling, qG/max(qCeiling,eps));
fprintf("  eval: Sharpe %.3f | success %d/%d | MaxDD P90 %.2f%% | TermW P10 %.0f\n", ...
    evalSharpe, succ, numEvalEpisodes, 100*evalMddP90, evalTermP10);
if qG < 3*qCeiling
    fprintf("  => Q BOUNDED at ~ceiling (IQL did NOT diverge where DQN did).\n");
else
    fprintf("  => Q still large (%.1fx ceiling) — investigate IQL loss/hyperparams.\n", qG/max(qCeiling,eps));
end

row = [seed, ME, EXPECTILE, TSF, LRATE, qG, qE, qL, vMean, margin, stateStd, effA, evalSharpe, succ, evalMddP90, evalTermP10];
writematrix(row, resultsCsv, "WriteMode", "append");
fprintf("  appended -> %s\n", resultsCsv);

%% Save IQL bundle for packaging/backtest (only when MODELPATH is set)
if exist('MODELPATH','var')
    FI = struct();
    FI.qNet = qNet;                 % greedy policy = argmax_a forward(qNet, obs)
    FI.W_frontier = W_frontier;
    FI.assetNames = assetNames;
    FI.macroMean = macroMean;  FI.macroStd = macroStd;
    FI.goalWealth = goalWealth;  FI.wealthClip = 5;
    FI.horizonPeriods = horizonPeriods;  FI.numActions = numActions;
    FI.meta = struct('learner','IQL (expectile 0.7, no OOD-max) — self-stabilizing', ...
        'seed',seed,'maxEpochs',ME,'expectile',EXPECTILE,'tau',TSF,'lr',LRATE, ...
        'state','[normWealth; timeFrac; DGS10_z; T10Y2Y_z; VIXCLS_z; DFF_z]', ...
        'note','greedy argmax forward(qNet); macro z-scored with FI.macroMean/Std; wealth clipped at 5x goal');
    md = fileparts(MODELPATH);  if ~isempty(md) && ~isfolder(md); mkdir(md); end
    save(MODELPATH, "FI", "-v7.3");
    fprintf("  saved IQL bundle -> %s\n", MODELPATH);
end


%% ===== local functions =====
function [lossV, grad] = vGrad(vNet, qTgt, s, aoh, expectile)
    qT = forward(qTgt, s);              % [numActions x B]
    qSA = sum(qT .* aoh, 1);           % [1 x B]  Q_target(s, a_dataset)
    v = forward(vNet, s);              % [1 x B]
    u = qSA - v;
    w = abs(expectile - single(extractdata(u) < 0));   % asymmetric expectile weight (constant)
    lossV = mean(w .* u.^2, 'all');
    grad = dlgradient(lossV, vNet.Learnables);
end

function [lossQ, grad] = qGrad(qNet, vNet, s, aoh, r, sp, dn, gamma)
    vSp = forward(vNet, sp);           % [1 x B]
    y = r + gamma .* (1 - dn) .* vSp;
    y = dlarray(extractdata(y));       % explicit stop-gradient: target is constant (smaller graph, clear intent)
    qAll = forward(qNet, s);           % [numActions x B]
    qSA = sum(qAll .* aoh, 1);         % [1 x B]
    lossQ = mean((qSA - y).^2, 'all');
    grad = dlgradient(lossQ, qNet.Learnables);
end

function [R, M_z] = localTestData(priceFile, macroFile, macroMean, macroStd)
    pricesTT = readtable(priceFile, 'VariableNamingRule','preserve');
    R = diff(log(pricesTT{:,2:end}));
    macroTT = readtable(macroFile, 'VariableNamingRule','preserve');
    M_z = (macroTT{:,2:end} - macroMean) ./ macroStd;
end
