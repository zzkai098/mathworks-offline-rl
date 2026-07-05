function [dsr, sr0, psr] = deflatedSharpe(rets, nTrials, trialSharpeVar)
%DEFLATEDSHARPE  Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
%   [dsr, sr0, psr] = deflatedSharpe(rets, nTrials, trialSharpeVar)
%
%   After ~10 strategy versions x 5 seeds x tuning, the best reported Sharpe
%   is subject to selection bias. The DSR discounts the observed Sharpe by
%   the Sharpe one would expect as the MAXIMUM across nTrials independent
%   trials under the null, then reports the probability the strategy's true
%   Sharpe still exceeds that benchmark.
%
%   rets           : column vector of per-period returns of the SELECTED strategy
%   nTrials        : number of trials/configurations examined (N)
%   trialSharpeVar : variance of the per-period (non-annualized) Sharpe
%                    estimates ACROSS the trials. If unknown, a common proxy
%                    is (1/T) i.e. the sampling variance of a single SR;
%                    pass [] to use that fallback.
%
%   Returns:
%     sr0 : expected maximum per-period Sharpe under the null across nTrials
%     dsr : deflated Sharpe ratio in [0,1] (prob. true SR > sr0)
%     psr : probabilistic Sharpe ratio vs a 0 benchmark (for reference)
%
%   All Sharpes here are PER-PERIOD (not annualized) so sr and sr0 share units.

    rets = rets(:);
    rets = rets(~isnan(rets));
    T = numel(rets);

    mu = mean(rets);
    sd = std(rets);
    if T < 4 || sd == 0
        dsr = NaN; sr0 = NaN; psr = NaN;
        return;
    end

    sr = mu / sd;                 % per-period Sharpe
    g3 = skewness(rets);          % skew of returns
    g4 = kurtosis(rets);          % raw kurtosis (normal = 3)

    if nargin < 3 || isempty(trialSharpeVar)
        trialSharpeVar = 1 / T;   % fallback: single-SR sampling variance
    end

    % Expected maximum Sharpe under the null across nTrials (Bailey-LdP)
    gamma = 0.5772156649015329;   % Euler-Mascheroni
    e = exp(1);
    nTrials = max(nTrials, 2);
    sr0 = sqrt(trialSharpeVar) * ( (1 - gamma) * norminv(1 - 1/nTrials) + ...
                                    gamma      * norminv(1 - 1/(nTrials * e)) );

    % PSR-style test statistic denominator (non-normal returns correction)
    denom = sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr^2);

    dsr = normcdf( (sr - sr0) * sqrt(T - 1) / denom );
    psr = normcdf( (sr - 0)   * sqrt(T - 1) / denom );
end
