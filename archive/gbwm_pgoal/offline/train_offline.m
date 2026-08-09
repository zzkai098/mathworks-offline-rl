function policyFn = train_offline(M, data, method, opts)
%TRAIN_OFFLINE  Offline RL on a fixed GBWM dataset: vanilla / CQL / IQL.
%   policyFn = train_offline(M, data, method, opts)
%
%   Small Q-network over state features [clip(W/goal,0,3), t/YEARS, regime],
%   NFRONT discrete frontier actions, pure terminal P(goal) reward, gamma=1
%   (finite horizon). Hand-written dlfeval loop (MATLAB's trainFromData can't
%   take a custom CQL/IQL loss). Returns a greedy policyFn(W,t,reg)->action for
%   gbwm.mcEval.
%
%   method : "vanilla" | "cql" | "iql"
%   opts   : struct with optional .steps .batch .lr .cqlAlpha .iqlTau .seed

if nargin < 4, opts = struct(); end
d = struct('steps',6000,'batch',256,'lr',1e-3,'cqlAlpha',1.0,'iqlTau',0.7,'seed',0);
fn = fieldnames(d);
for i = 1:numel(fn), if ~isfield(opts,fn{i}), opts.(fn{i}) = d.(fn{i}); end, end
rng(opts.seed, 'twister');
nA = M.NFRONT; goal = M.YEARS; %#ok<NASGU>
goal = M.goal;

% --- featurize dataset ---
feat = @(W,t,reg) [min(max(W(:)'/goal,0),3); t(:)'/M.YEARS; reg(:)'];   % 3 x N
S  = feat(data.W,  data.t,           data.reg);
SN = feat(data.Wn, min(data.t+1,M.YEARS), data.regn);
A  = double(data.a(:)');           % 1 x N
R  = double(data.r(:)');
D  = double(data.done(:)');
N  = numel(A);

% --- networks ---
qNet = makeNet(nA); qtNet = qNet;
avgG = []; avgSqG = [];
if method == "iql"
    vNet = makeNet(1); avgGv = []; avgSqGv = [];
end

gamma = 1.0;
for step = 1:opts.steps
    idx = randi(N, [1 opts.batch]);
    bS = dlarray(S(:,idx),'CB'); bSN = dlarray(SN(:,idx),'CB');
    bA = A(idx); bR = dlarray(R(idx),'CB'); bD = dlarray(D(idx),'CB');
    Aoh = zeros(nA, opts.batch); Aoh(sub2ind([nA opts.batch], bA, 1:opts.batch)) = 1;
    bAoh = dlarray(Aoh,'CB');

    if method == "iql"
        [gv] = dlfeval(@iqlVGrad, vNet, qtNet, bS, bAoh, opts.iqlTau);
        [vNet, avgGv, avgSqGv] = adamupdate(vNet, gv, avgGv, avgSqGv, step, opts.lr);
        [gq] = dlfeval(@iqlQGrad, qNet, vNet, bS, bAoh, bR, bSN, bD, gamma);
        [qNet, avgG, avgSqG] = adamupdate(qNet, gq, avgG, avgSqG, step, opts.lr);
    else
        [gq] = dlfeval(@dqnGrad, qNet, qtNet, bS, bAoh, bR, bSN, bD, gamma, ...
                       method=="cql", opts.cqlAlpha);
        [qNet, avgG, avgSqG] = adamupdate(qNet, gq, avgG, avgSqG, step, opts.lr);
    end
    % soft target update
    qtNet.Learnables = dlupdate(@(t,s) 0.995*t + 0.005*s, qtNet.Learnables, qNet.Learnables);
end

policyFn = @(W,t,reg) greedy(qNet, feat(W, repmat(t,size(W)), reg));
end


function net = makeNet(nout)
    layers = [
        featureInputLayer(3,'Name','in')
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(nout,'Name','out')];
    net = dlnetwork(layerGraph(layers));
end

function a = greedy(qNet, F)
    Q = predict(qNet, dlarray(F,'CB'));
    [~, a] = max(extractdata(Q), [], 1);
    a = a(:);
end

function g = dqnGrad(qNet, qtNet, S, Aoh, R, SN, D, gamma, isCQL, alpha)
    Q = forward(qNet, S);
    Qsa = sum(Q .* Aoh, 1);
    Qn = forward(qtNet, SN);                    % constant wrt qNet.Learnables
    y = R + gamma .* (1 - D) .* max(Qn, [], 1);
    td = mean((Qsa - y).^2, 'all');
    if isCQL
        m = max(Q, [], 1);
        lse = m + log(sum(exp(Q - m), 1));
        cql = mean(lse - Qsa, 'all');
        loss = td + alpha * cql;
    else
        loss = td;
    end
    g = dlgradient(loss, qNet.Learnables);
end

function gv = iqlVGrad(vNet, qtNet, S, Aoh, tau)
    QtSA = sum(forward(qtNet, S) .* Aoh, 1);    % target Q at data action (const)
    Vs = forward(vNet, S);
    diff = QtSA - Vs;
    w = tau * (diff > 0) + (1 - tau) * (diff <= 0);
    loss = mean(w .* diff.^2, 'all');
    gv = dlgradient(loss, vNet.Learnables);
end

function gq = iqlQGrad(qNet, vNet, S, Aoh, R, SN, D, gamma)
    Q = forward(qNet, S);
    Qsa = sum(Q .* Aoh, 1);
    Vsn = forward(vNet, SN);                     % const wrt qNet.Learnables
    y = R + gamma .* (1 - D) .* Vsn;
    loss = mean((Qsa - y).^2, 'all');
    gq = dlgradient(loss, qNet.Learnables);
end
