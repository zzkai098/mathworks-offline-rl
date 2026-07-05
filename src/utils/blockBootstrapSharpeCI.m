function [ciLow, ciHigh, sharpeHat, bootSharpes] = ...
        blockBootstrapSharpeCI(rets, blockLen, nBoot, annualizeFactor, alphaLevel)
%BLOCKBOOTSTRAPSHARPECI  Moving-block bootstrap CI for the Sharpe ratio.
%   [ciLow, ciHigh, sharpeHat, bootSharpes] = ...
%       blockBootstrapSharpeCI(rets, blockLen, nBoot, annualizeFactor, alphaLevel)
%
%   The project's eval windows are 30-day and OVERLAPPING, so per-return
%   observations are autocorrelated and a naive (iid) bootstrap understates
%   the Sharpe standard error. A moving-block bootstrap with block length ~
%   the horizon preserves within-block dependence.
%
%   rets            : column vector of per-period (log) portfolio returns
%   blockLen        : block length (default 30, ~= horizon)
%   nBoot           : number of bootstrap resamples (default 2000)
%   annualizeFactor : Sharpe scaling (default sqrt(252))
%   alphaLevel      : two-sided level (default 0.05 -> 95% CI)
%
%   Returns the [alpha/2, 1-alpha/2] percentile CI on the annualized Sharpe,
%   the point estimate, and the raw bootstrap distribution.

    if nargin < 2 || isempty(blockLen),        blockLen = 30;          end
    if nargin < 3 || isempty(nBoot),           nBoot = 2000;           end
    if nargin < 4 || isempty(annualizeFactor), annualizeFactor = sqrt(252); end
    if nargin < 5 || isempty(alphaLevel),      alphaLevel = 0.05;      end

    rets = rets(:);
    rets = rets(~isnan(rets));
    n = numel(rets);

    sdAll = std(rets);
    if n < 2 || sdAll == 0
        sharpeHat = NaN; ciLow = NaN; ciHigh = NaN;
        bootSharpes = nan(nBoot, 1);
        return;
    end
    sharpeHat = mean(rets) / sdAll * annualizeFactor;

    blockLen = min(blockLen, n);
    nBlocks  = ceil(n / blockLen);
    maxStart = n - blockLen + 1;

    bootSharpes = nan(nBoot, 1);
    for b = 1:nBoot
        starts = randi(maxStart, nBlocks, 1);
        idx = zeros(nBlocks * blockLen, 1);
        for k = 1:nBlocks
            idx((k-1)*blockLen + (1:blockLen)) = starts(k):(starts(k) + blockLen - 1);
        end
        idx = idx(1:n);                 % trim to original length
        rb  = rets(idx);
        sdb = std(rb);
        if sdb > 0
            bootSharpes(b) = mean(rb) / sdb * annualizeFactor;
        end
    end

    ciLow  = prctile(bootSharpes, 100 * alphaLevel / 2);
    ciHigh = prctile(bootSharpes, 100 * (1 - alphaLevel / 2));
end
