%% GBWM P(goal) — PRE-FLIGHT (no RL), MATLAB
% HARD-STOP gate before any RL: prove the 10-year annual GBWM game is *winnable
% by dynamics*, on a regime-switching return model fit on TRAIN (2010-2019) only:
%
%   regime-aware DP  >  regime-agnostic DP  >  best static mix  >  60/40
%
% with the goal calibrated to a STRETCH (DP-optimal P(goal) ~= 0.75) so the game
% sits in the "reach-y" regime where wealth-dependent risk management matters
% (calibrating to 60/40=60% instead SATURATES all risky policies ~0.97 and hides
% the dynamic edge — verified). Objective = pure P(goal) = P(W_T >= goal).
%
% Mirrors the validated Python prototype in ../_prototype_python/ (goal ~1928,
% ceiling ~0.75, dynamics margin +0.125). All fit on train only; no leakage.
%
% Run from repo root:  matlab -batch "addpath('gbwm_pgoal/dp'); preflight"

clear; clc;

%% GBWM instance
YEARS   = 10;      W0 = 100;   CONTRIB = 10;
NFRONT  = 15;      GHN = 15;
WGRID_N = 500;     WGRID_MAX_MULT = 12;
VIX_THR_Z = 1.5;   SLOPE_THR_Z = -1.5;   MONTH_STRESS_FRAC = 0.25;
STRETCH = 0.75;    NMC = 200000;
COL_SLOPE = 2;     COL_VIX = 3;          % macro cols: DGS10,T10Y2Y,VIXCLS,DFF
EQUITY = ["AAPL","MSFT","NVDA","JPM","BAC","V","JNJ","UNH","XOM","CVX","PG","KO","CAT"];
BOND = "TLT";

%% ---- data -> monthly returns + monthly regime (train only) ----
P = readtable("data/prices_train.csv");
M = readtable("data/macro_train.csv");
dates   = datetime(string(P.Date));
assets  = string(P.Properties.VariableNames(2:end));
px      = P{:, 2:end};
macro   = M{:, 2:end};

% daily regime (train-z-scored macro; mentor's gate)
zc = (macro - mean(macro)) ./ std(macro);
stressDay = (zc(:, COL_VIX) > VIX_THR_Z) | (zc(:, COL_SLOPE) < SLOPE_THR_Z);

% month-end prices -> monthly log returns; monthly regime = >=25% stress days
ym = dateshift(dates, 'start', 'month');
[umon, ~, g] = unique(ym);
nM = numel(umon);
mEndPx = zeros(nM, numel(assets));
mStress = false(nM, 1);
for k = 1:nM
    idx = find(g == k);
    mEndPx(k, :) = px(idx(end), :);
    mStress(k) = mean(stressDay(idx)) >= MONTH_STRESS_FRAC;
end
mret = diff(log(mEndPx));           % (nM-1 x nAssets)
mStress = mStress(2:end);           % align to returns
fprintf("[data] %d months, stress-month rate = %.1f%%\n", size(mret,1), 100*mean(mStress));

%% ---- efficient frontier (unconditional monthly mu/Sigma) ----
mu = mean(mret)';   Sigma = cov(mret);
p = Portfolio('AssetList', cellstr(assets));
p = setDefaultConstraints(p);
p = setAssetMoments(p, mu, Sigma);
Wfront = estimateFrontier(p, NFRONT);      % nAssets x NFRONT

% 60/40 weight
w6040 = zeros(numel(assets), 1);
for i = 1:numel(assets)
    if any(assets(i) == EQUITY), w6040(i) = 0.60/numel(EQUITY);
    elseif assets(i) == BOND,   w6040(i) = 0.40; end
end
Wall = [Wfront, w6040];                     % nAssets x (NFRONT+1)

% regime-conditional ANNUAL params per portfolio: annual ~ N(12*m, 12*s^2)
am = zeros(size(Wall,2), 2); as = zeros(size(Wall,2), 2);
for a = 1:size(Wall,2)
    r = mret * Wall(:, a);
    for rg = 1:2
        mask = (mStress == (rg-1));
        am(a, rg) = 12 * mean(r(mask));
        as(a, rg) = sqrt(12) * std(r(mask), 0);
    end
end
I6040 = NFRONT + 1;
amF = am(1:NFRONT, :); asF = as(1:NFRONT, :);

% annual regime transition (Laplace-smoothed monthly ^12) + stationary
Pm = ones(2,2);
s = double(mStress);
for k = 1:numel(s)-1, Pm(s(k)+1, s(k+1)+1) = Pm(s(k)+1, s(k+1)+1) + 1; end
Pm = Pm ./ sum(Pm, 2);
Pann = Pm^12;
[V, D] = eig(Pann');  [~, j] = min(abs(diag(D) - 1));
stat = real(V(:, j)); stat = stat / sum(stat);
fprintf("[regime] annual P = [%.3f %.3f; %.3f %.3f]  stationary = [%.3f %.3f]\n", ...
    Pann(1,1),Pann(1,2),Pann(2,1),Pann(2,2), stat(1), stat(2));

% Gauss-Hermite nodes (Golub-Welsch): E_{N(0,1)}[f] = sum ghw .* f(ghz)
[ghz, ghw] = ghquad(GHN);

%% ---- calibrate STRETCH goal: bisection on DP-aware P(goal) ~= 0.75 ----
lo = W0 + CONTRIB*YEARS; hi = 8000;
for it = 1:18
    mid = 0.5*(lo+hi);
    [Wg, pol] = solveDP(mid, amF, asF, Pann, stat, false, ghz, ghw, YEARS, CONTRIB, WGRID_N, WGRID_MAX_MULT);
    pa = mcEval(@(W,t,rg) polLookup(pol, Wg, W, t, rg, false), amF, asF, Pm, ...
                mid, 50000, YEARS, CONTRIB, W0, 1);
    if pa > STRETCH, lo = mid; else, hi = mid; end
end
goal = 0.5*(lo+hi);
fprintf("[calib] goal = %.1f  (DP-aware ceiling ~= %.2f)\n", goal, STRETCH);

%% ---- solve DPs + evaluate everything under the TRUE model ----
[WgA, polA] = solveDP(goal, amF, asF, Pann, stat, false, ghz, ghw, YEARS, CONTRIB, WGRID_N, WGRID_MAX_MULT);
[WgN, polN] = solveDP(goal, amF, asF, Pann, stat, true,  ghz, ghw, YEARS, CONTRIB, WGRID_N, WGRID_MAX_MULT);

pAware = mcEval(@(W,t,rg) polLookup(polA, WgA, W, t, rg, false), amF, asF, Pm, goal, NMC, YEARS, CONTRIB, W0, 10);
pAgn   = mcEval(@(W,t,rg) polLookup(polN, WgN, W, t, rg, true),  amF, asF, Pm, goal, NMC, YEARS, CONTRIB, W0, 11);
p6040  = mcEval(@(W,t,rg) repmat(I6040, size(W)),                am,  as,  Pm, goal, NMC, YEARS, CONTRIB, W0, 12);

% best static single frontier portfolio
statP = zeros(NFRONT,1);
for a = 1:NFRONT
    statP(a) = mcEval(@(W,t,rg) repmat(a, size(W)), amF, asF, Pm, goal, NMC, YEARS, CONTRIB, W0, 100+a);
end
[bStat, bStatA] = max(statP);

% best linear glide path (search start,end frontier index)
bGlide = -1;
for i0 = 1:NFRONT
    for i1 = 1:NFRONT
        sched = round(linspace(i0, i1, YEARS));
        pg = mcEval(@(W,t,rg) repmat(sched(t), size(W)), amF, asF, Pm, goal, NMC, YEARS, CONTRIB, W0, 500);
        if pg > bGlide, bGlide = pg; end
    end
end

%% ---- report ----
fprintf("\n============ PRE-FLIGHT P(goal) (MC under true regime model, N=%d) ============\n", NMC);
fprintf("  regime-aware DP (ceiling) : %.3f\n", pAware);
fprintf("  regime-agnostic DP        : %.3f\n", pAgn);
fprintf("  best glide path           : %.3f\n", bGlide);
fprintf("  best static (#%d)          : %.3f\n", bStatA, bStat);
fprintf("  60/40                     : %.3f\n", p6040);
fprintf("\n---- attribution margins (the REAL go/no-go) ----\n");
fprintf("  regime signal   aware - agnostic : %+.3f\n", pAware - pAgn);
fprintf("  dynamics        agnostic - static: %+.3f\n", pAgn - bStat);
fprintf("  risk level      static - 60/40   : %+.3f\n", bStat - p6040);
dynMargin = pAware - bStat;
fprintf("  >>> dynamics+regime over best-static (aware - static) : %+.3f\n", dynMargin);
ok = (pAware > pAgn) && (pAgn > bStat) && (bStat > p6040) && (dynMargin >= 0.03);
if ok
    fprintf("\n  VERDICT: PASS -> dynamics adds real value, proceed to RL\n");
else
    fprintf("\n  VERDICT: WEAK -> dynamic edge < 0.03, reconsider instance\n");
end


%% ================= local functions =================
function [x, w] = ghquad(n)
    % Golub-Welsch Gauss-Hermite; returns nodes/weights s.t.
    % E_{z~N(0,1)}[f(z)] = sum w .* f(x).
    i = 1:n-1;  b = sqrt(i/2);
    J = diag(b, 1) + diag(b, -1);
    [V, D] = eig(J);
    [xh, idx] = sort(diag(D));
    wh = (V(1, idx).^2)' * sqrt(pi);       % physicists' GH weights
    x = (sqrt(2) * xh)';                    % -> N(0,1) nodes
    w = (wh / sqrt(pi))';
end

function [Wgrid, pol] = solveDP(goal, am, as, Pann, stat, agnostic, ghz, ghw, YEARS, CONTRIB, nW, maxMult)
    Wgrid = linspace(1, maxMult*goal, nW);
    nA = size(am, 1);
    Vterm = double(Wgrid >= goal);
    if agnostic
        V = Vterm; pol = cell(YEARS,1);
        for t = YEARS:-1:1
            Q = zeros(nA, nW);
            for a = 1:nA
                ev = zeros(nW,1);
                for rg = 1:2
                    ev = ev + stat(rg) * expectV(Wgrid, V, am(a,rg), as(a,rg), ghz, ghw, CONTRIB);
                end
                Q(a, :) = ev';
            end
            [V, ai] = max(Q, [], 1);
            pol{t} = ai;
        end
    else
        V = [Vterm; Vterm]; pol = cell(YEARS,1);   % rows = regime
        for t = YEARS:-1:1
            newV = zeros(2, nW); newPi = zeros(2, nW);
            for rg = 1:2
                Q = zeros(nA, nW);
                for a = 1:nA
                    ev = zeros(nW,1);
                    for rp = 1:2
                        ev = ev + Pann(rg,rp) * expectV(Wgrid, V(rp,:), am(a,rg), as(a,rg), ghz, ghw, CONTRIB);
                    end
                    Q(a, :) = ev';
                end
                [mx, ai] = max(Q, [], 1);
                newV(rg, :) = mx; newPi(rg, :) = ai;
            end
            V = newV; pol{t} = newPi;
        end
    end
end

function ev = expectV(Wgrid, Vnext, m, s, ghz, ghw, CONTRIB)
    % E[ Vnext( (W+C)*exp(m + s*z) ) ] over z-nodes, for each W in Wgrid (col vec)
    Wp = (Wgrid' + CONTRIB) .* exp(m + s .* ghz);          % nW x nNodes
    Wp = min(max(Wp, Wgrid(1)), Wgrid(end));               % clamp (V flat outside)
    Vq = interp1(Wgrid, Vnext, Wp, 'linear');              % nW x nNodes
    ev = sum(Vq .* ghw, 2);                                % nW x 1
end

function a = polLookup(pol, Wgrid, W, t, rg, agnostic)
    idx = min(max(round(interp1(Wgrid, 1:numel(Wgrid), W, 'nearest', 'extrap')), 1), numel(Wgrid));
    if agnostic
        pt = pol{t};  a = pt(idx);
    else
        pt = pol{t};                 % 2 x nW
        a = zeros(size(W));
        a(rg==0) = pt(1, idx(rg==0));
        a(rg==1) = pt(2, idx(rg==1));
    end
end

function p = mcEval(policyFn, am, as, Pm, goal, nmc, YEARS, CONTRIB, W0, seed)
    rng(seed, 'twister');
    W = W0 * ones(nmc, 1);  reg = zeros(nmc, 1);
    for t = 1:YEARS
        a = policyFn(W, t, reg);
        a = a(:);
        lin = a + reg*size(am,1);          % index into am(a,reg) flattened? build directly
        m = am(sub2ind(size(am), a, reg+1));
        sd = as(sub2ind(size(as), a, reg+1));
        z = randn(nmc, 1);
        W = (W + CONTRIB) .* exp(m + sd .* z);
        reg = double(rand(nmc,1) < Pm(sub2ind(size(Pm), reg+1, 2*ones(nmc,1))));
    end
    p = mean(W >= goal);
end
