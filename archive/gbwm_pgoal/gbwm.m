classdef gbwm
%GBWM  Shared library for the GBWM P(goal) offline-RL study (MATLAB).
%   Single source of truth for the regime-switching return model, the
%   efficient-frontier action set, the DP optimum (the measurable CEILING),
%   Monte-Carlo P(goal) evaluation, and offline-episode logging. Fit on TRAIN
%   (2010-2019) only. Validated against the Python prototype in
%   _prototype_python/ (goal ~1949, DP ceiling ~0.75, dynamics margin +0.125).
%
%   Usage:
%       M = gbwm.model();                        % build + calibrate + DP ceiling
%       p = gbwm.mcEval(M, policyFn, M.goal);    % MC P(goal) of any policy
%       d = gbwm.logEpisodes(M, "random", 500);  % offline dataset
%
%   State the RL agent and the DP share: (normWealth, timeFraction, regime).
%   Objective (primary): pure P(goal) = P(W_T >= goal).

methods (Static)

    function M = returnModel()
        % --- constants ---
        M.YEARS = 10; M.W0 = 100; M.CONTRIB = 10;
        M.NFRONT = 15; M.GHN = 15;
        M.WGRID_N = 500; M.WGRID_MAX_MULT = 12;
        VIX_THR_Z = 1.5; SLOPE_THR_Z = -1.5; MONTH_STRESS_FRAC = 0.25;
        COL_SLOPE = 2; COL_VIX = 3;
        EQUITY = ["AAPL","MSFT","NVDA","JPM","BAC","V","JNJ","UNH","XOM","CVX","PG","KO","CAT"];
        BOND = "TLT";

        % --- data -> monthly returns + monthly regime (train only) ---
        P = readtable("data/prices_train.csv");
        Mm = readtable("data/macro_train.csv");
        dates  = datetime(string(P.Date));
        assets = string(P.Properties.VariableNames(2:end));
        px = P{:, 2:end}; macro = Mm{:, 2:end};

        zc = (macro - mean(macro)) ./ std(macro);
        stressDay = (zc(:, COL_VIX) > VIX_THR_Z) | (zc(:, COL_SLOPE) < SLOPE_THR_Z);

        ym = dateshift(dates, 'start', 'month');
        [~, ~, g] = unique(ym);
        nM = max(g);
        mEndPx = zeros(nM, numel(assets)); mStress = false(nM, 1);
        for k = 1:nM
            idx = find(g == k);
            mEndPx(k, :) = px(idx(end), :);
            mStress(k) = mean(stressDay(idx)) >= MONTH_STRESS_FRAC;
        end
        mret = diff(log(mEndPx)); mStress = mStress(2:end);
        M.mStressRate = mean(mStress);

        % --- efficient frontier (unconditional monthly mu/Sigma) ---
        mu = mean(mret)'; Sigma = cov(mret);
        p = Portfolio('AssetList', cellstr(assets));
        p = setDefaultConstraints(p); p = setAssetMoments(p, mu, Sigma);
        Wfront = estimateFrontier(p, M.NFRONT);

        w6040 = zeros(numel(assets), 1);
        for i = 1:numel(assets)
            if any(assets(i) == EQUITY), w6040(i) = 0.60/numel(EQUITY);
            elseif assets(i) == BOND,   w6040(i) = 0.40; end
        end
        Wall = [Wfront, w6040];

        % regime-conditional ANNUAL params: annual ~ N(12*m, 12*s^2)
        na = size(Wall, 2);
        am = zeros(na, 2); as = zeros(na, 2);
        for a = 1:na
            r = mret * Wall(:, a);
            for rg = 1:2
                mask = (mStress == (rg-1));
                am(a, rg) = 12 * mean(r(mask));
                as(a, rg) = sqrt(12) * std(r(mask), 0);
            end
        end
        M.am = am; M.as = as;                       % includes 60/40 at end
        M.amF = am(1:M.NFRONT, :); M.asF = as(1:M.NFRONT, :);
        M.I6040 = M.NFRONT + 1;
        M.Wfront = Wfront; M.assets = assets;

        % annual regime transition + stationary
        Pm = ones(2, 2); s = double(mStress);
        for k = 1:numel(s)-1, Pm(s(k)+1, s(k+1)+1) = Pm(s(k)+1, s(k+1)+1) + 1; end
        Pm = Pm ./ sum(Pm, 2);
        M.Pm = Pm; M.Pann = Pm^12;
        [V, D] = eig(M.Pann'); [~, j] = min(abs(diag(D) - 1));
        stat = real(V(:, j)); M.stat = stat / sum(stat);

        [M.ghz, M.ghw] = gbwm.ghquad(M.GHN);
    end

    function M = model(verbose)
        if nargin < 1, verbose = true; end
        M = gbwm.returnModel();
        STRETCH = 0.75;
        lo = M.W0 + M.CONTRIB*M.YEARS; hi = 8000;
        for it = 1:18
            mid = 0.5*(lo+hi);
            [Wg, pol] = gbwm.solveDP(M, mid, false);
            pf = @(W,t,rg) gbwm.polLookup(pol, Wg, W, t, rg, false);
            pa = gbwm.mcEval(M, pf, mid, 50000, 1);
            if pa > STRETCH, lo = mid; else, hi = mid; end
        end
        M.goal = 0.5*(lo+hi);
        [M.dp_Wgrid, M.dp_pol] = gbwm.solveDP(M, M.goal, false);
        M.dp_policyFn = @(W,t,rg) gbwm.polLookup(M.dp_pol, M.dp_Wgrid, W, t, rg, false);
        M.dp_ceiling = gbwm.mcEval(M, M.dp_policyFn, M.goal, 200000, 2);
        if verbose
            fprintf("[gbwm] goal=%.1f dp_ceiling=%.3f stress_rate=%.1f%%\n", ...
                M.goal, M.dp_ceiling, 100*M.mStressRate);
        end
    end

    function [Wgrid, pol] = solveDP(M, goal, agnostic)
        Wgrid = linspace(1, M.WGRID_MAX_MULT*goal, M.WGRID_N);
        am = M.amF; as = M.asF; nA = size(am, 1);
        Vterm = double(Wgrid >= goal);
        if agnostic
            V = Vterm; pol = cell(M.YEARS, 1);
            for t = M.YEARS:-1:1
                Q = zeros(nA, M.WGRID_N);
                for a = 1:nA
                    ev = zeros(M.WGRID_N, 1);
                    for rg = 1:2
                        ev = ev + M.stat(rg)*gbwm.expectV(Wgrid, V, am(a,rg), as(a,rg), M.ghz, M.ghw, M.CONTRIB);
                    end
                    Q(a, :) = ev';
                end
                [V, ai] = max(Q, [], 1); pol{t} = ai;
            end
        else
            V = [Vterm; Vterm]; pol = cell(M.YEARS, 1);
            for t = M.YEARS:-1:1
                newV = zeros(2, M.WGRID_N); newPi = zeros(2, M.WGRID_N);
                for rg = 1:2
                    Q = zeros(nA, M.WGRID_N);
                    for a = 1:nA
                        ev = zeros(M.WGRID_N, 1);
                        for rp = 1:2
                            ev = ev + M.Pann(rg,rp)*gbwm.expectV(Wgrid, V(rp,:), am(a,rg), as(a,rg), M.ghz, M.ghw, M.CONTRIB);
                        end
                        Q(a, :) = ev';
                    end
                    [mx, ai] = max(Q, [], 1); newV(rg, :) = mx; newPi(rg, :) = ai;
                end
                V = newV; pol{t} = newPi;
            end
        end
    end

    function ev = expectV(Wgrid, Vnext, m, s, ghz, ghw, CONTRIB)
        Wp = (Wgrid' + CONTRIB) .* exp(m + s .* ghz);
        Wp = min(max(Wp, Wgrid(1)), Wgrid(end));
        ev = sum(interp1(Wgrid, Vnext, Wp, 'linear') .* ghw, 2);
    end

    function a = polLookup(pol, Wgrid, W, t, rg, agnostic)
        idx = min(max(round(interp1(Wgrid, 1:numel(Wgrid), W, 'nearest', 'extrap')), 1), numel(Wgrid));
        pt = pol{t};
        if agnostic
            a = pt(idx)';
        else
            a = zeros(size(W));
            a(rg==0) = pt(1, idx(rg==0));
            a(rg==1) = pt(2, idx(rg==1));
        end
        a = a(:);
    end

    function p = mcEval(M, policyFn, goal, nmc, seed)
        if nargin < 4, nmc = 200000; end
        if nargin < 5, seed = 0; end
        rng(seed, 'twister');
        am = M.am; as = M.as;
        W = M.W0*ones(nmc,1); reg = zeros(nmc,1);
        for t = 1:M.YEARS
            a = policyFn(W, t, reg); a = a(:);
            m  = am(sub2ind(size(am), a, reg+1));
            sd = as(sub2ind(size(as), a, reg+1));
            W = (W + M.CONTRIB) .* exp(m + sd .* randn(nmc,1));
            reg = double(rand(nmc,1) < M.Pm(sub2ind(size(M.Pm), reg+1, 2*ones(nmc,1))));
        end
        p = mean(W >= goal);
    end

    function s = mcEvalFull(M, policyFn, goal, nmc, seed)
        % Like mcEval but also returns path tail metrics for the drawdown variant.
        % s.pGoal, s.meanMaxDD, s.p90MaxDD, s.termP10.
        if nargin < 4, nmc = 200000; end
        if nargin < 5, seed = 0; end
        rng(seed, 'twister');
        am = M.am; as = M.as;
        W = M.W0*ones(nmc,1); reg = zeros(nmc,1);
        peak = W; maxDD = zeros(nmc,1);
        for t = 1:M.YEARS
            a = policyFn(W, t, reg); a = a(:);
            m  = am(sub2ind(size(am), a, reg+1));
            sd = as(sub2ind(size(as), a, reg+1));
            W = (W + M.CONTRIB) .* exp(m + sd .* randn(nmc,1));
            peak = max(peak, W);
            maxDD = max(maxDD, (peak - W) ./ peak);
            reg = double(rand(nmc,1) < M.Pm(sub2ind(size(M.Pm), reg+1, 2*ones(nmc,1))));
        end
        s.pGoal = mean(W >= goal);
        s.meanMaxDD = mean(maxDD);
        s.p90MaxDD = prctile(maxDD, 90);
        s.termP10 = prctile(W, 10);
    end

    function d = logEpisodes(M, behavior, nEpisodes, seed, useDrawdown)
        % Offline transitions under a SUBOPTIMAL behavior policy.
        %   useDrawdown=false (default): pure terminal P(goal) reward (primary study).
        %   useDrawdown=true: mentor's regime-gated drawdown penalty added per step
        %     (6.1 shaping: beta_stress=8, beta_normal=2, DD_THR=0.03 on the
        %     within-episode running drawdown). Objective is then P(goal)+tail, so
        %     compare on P(goal) AND tail (not "pure gap-to-DP").
        % behavior: "random" | "noisy_glide".
        if nargin < 4, seed = 0; end
        if nargin < 5, useDrawdown = false; end
        BETA_HI = 8; BETA_LO = 2; DD_THR = 0.03;
        rng(seed, 'twister');
        amF = M.amF; asF = M.asF; nA = M.NFRONT;
        glide = round(linspace(nA, round(nA/2), M.YEARS));   % aggressive -> mid
        N = nEpisodes * M.YEARS;
        W = zeros(N,1); t = zeros(N,1); reg = zeros(N,1); a = zeros(N,1);
        r = zeros(N,1); Wn = zeros(N,1); regn = zeros(N,1); done = false(N,1);
        i = 0;
        for ep = 1:nEpisodes
            w = M.W0; rg = 0; peak = M.W0;
            for tt = 1:M.YEARS
                if behavior == "random"
                    act = randi(nA);
                else
                    if rand > 0.3, act = glide(tt); else, act = randi(nA); end
                end
                z = randn;
                wn = (w + M.CONTRIB) * exp(amF(act, rg+1) + asF(act, rg+1)*z);
                rgn = double(rand < M.Pm(rg+1, 2));
                dn = (tt == M.YEARS);
                rew = 0; if dn, rew = double(wn >= M.goal); end
                if useDrawdown
                    peak = max(peak, wn);
                    DD = (peak - wn) / peak;
                    beta = BETA_LO; if rg == 1, beta = BETA_HI; end   % gate on CURRENT regime
                    rew = rew - beta * max(0, DD - DD_THR);
                end
                i = i + 1;
                W(i)=w; t(i)=tt; reg(i)=rg; a(i)=act; r(i)=rew;
                Wn(i)=wn; regn(i)=rgn; done(i)=dn;
                w = wn; rg = rgn;
            end
        end
        d = struct('W',W,'t',t,'reg',reg,'a',a,'r',r,'Wn',Wn,'regn',regn,'done',done);
    end

    function [x, w] = ghquad(n)
        i = 1:n-1; b = sqrt(i/2);
        J = diag(b, 1) + diag(b, -1);
        [V, D] = eig(J); [xh, idx] = sort(diag(D));
        wh = (V(1, idx).^2)' * sqrt(pi);
        x = (sqrt(2) * xh)'; w = (wh / sqrt(pi))';
    end

end
end
