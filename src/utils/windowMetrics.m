function m = windowMetrics(G, T, costBps, basis, doStats)
%WINDOWMETRICS  Descriptive + inferential metrics for one policy's replay.
%   m = windowMetrics(G, T, costBps, basis, doStats)
%
%   G,T come from replayPolicy; costBps is a scalar turnover cost (basis points)
%   charged on the drift-aware turnover. Returns a struct m with:
%     .totRet        chained net total return across all windows
%     .curve         chained net wealth curve (length = #windows*H + 1)
%     .chainedSharpe annualized Sharpe of the single chained net-return series (HONEST
%                    primary metric — no per-window reset, so drawdowns compound)
%     .meanSharpe    mean of per-window annualized Sharpe (window-reset; flatters)
%     .maxDDp90      90th-percentile per-window MaxDD
%     .worstSharpe   min per-window Sharpe
%     .termP10       10th-percentile per-window terminal wealth
%     .meanTerm      mean per-window terminal wealth
%     .success       # windows reaching goalWealth
%     .turnover      mean per-step turnover
%   If doStats (default false), also:
%     .ci  [lo hi]   block-bootstrap 95%% CI for the chained Sharpe (block = horizon)
%     .dsr           Deflated Sharpe (Bailey & Lopez de Prado) on the chained series
%   CI/DSR are on ONE net-return path (a single agent/seed or one ensemble), never
%   pooled across seeds — pooling the same market path would fabricate independence.

    if nargin < 5 || isempty(doStats); doStats = false; end
    c  = costBps/1e4;  af = basis.annualizeFactor;
    iw = basis.initialWealth;  gw = basis.goalWealth;

    n  = size(G,1);
    sh = nan(n,1);  dd = nan(n,1);  tw = nan(n,1);  sc = nan(n,1);  tn = nan(n,1);
    chainedNet = [];
    for ep = 1:n
        g = G(ep,:)';  tu = T(ep,:)';
        if any(isnan(g)); continue; end
        nr = g - c*tu;                       % per-step net return
        w  = iw*[1; cumprod(1+nr)];  ln = diff(log(w));
        if std(ln) == 0; sh(ep) = NaN; else; sh(ep) = mean(ln)/std(ln)*af; end  % guard flat window

        dd(ep) = max(cummax(w) - w)/max(w);
        tw(ep) = w(end);  sc(ep) = double(w(end) >= gw);  tn(ep) = mean(tu);
        chainedNet = [chainedNet; nr];       %#ok<AGROW>
    end

    m.meanSharpe = mean(sh,'omitnan');
    m.maxDDp90   = prctile(dd,90);
    m.worstSharpe= min(sh);
    m.termP10    = prctile(tw,10);
    m.meanTerm   = mean(tw,'omitnan');
    m.success    = sum(sc,'omitnan');
    m.turnover   = mean(tn,'omitnan');
    m.meanMaxDD  = mean(dd,'omitnan');
    m.perWin     = struct('sharpe',sh,'maxDD',dd,'termW',tw,'success',sc);   % for cross-seed pooling

    curve = iw*[1; cumprod(1+chainedNet)];
    m.curve  = curve;
    m.totRet = curve(end)/curve(1) - 1;
    lnC = diff(log(curve));
    m.chainedSharpe = mean(lnC)/std(lnC)*af;

    if doStats
        addpath(fullfile("src","utils"));   % blockBootstrapSharpeCI, deflatedSharpe
        % Feed the SAME log-return series used for m.chainedSharpe, so the point
        % estimate and its CI/DSR estimate the same quantity (matches week10_backtest).
        [lo,hi] = blockBootstrapSharpeCI(lnC, basis.horizonPeriods, 2000, af);
        m.ci  = [lo hi];
        m.dsr = deflatedSharpe(lnC, 40, []);   % 40 = honest trial count (week4-10)
    end
end
