%% Data Exploration — SimulatedData.xlsx
% Week 3, Task 2: understand the simulated dataset.
% This script characterizes the data so we know what assumptions the
% downstream RL pipeline is implicitly making.

clear; clc; close all;

%% Load raw data and inspect structure

pricesTT = readtable("data/SimulatedData.xlsx");
fprintf("=== Raw table structure ===\n");
fprintf("Size: %d rows x %d columns\n", height(pricesTT), width(pricesTT));
fprintf("Column names: %s\n", strjoin(pricesTT.Properties.VariableNames, ", "));
fprintf("Column types:\n"); disp(varfun(@class, pricesTT, 'OutputFormat','table'));

head(pricesTT, 5)

%% Identify price columns
% From earlier inspection: the last 3 columns are Stock1/2/3 prices.
% Auto-detect numeric columns to be robust.

isNumCol = varfun(@(x) isnumeric(x), pricesTT, 'OutputFormat','uniform');
priceCols = pricesTT.Properties.VariableNames(isNumCol);
fprintf("\nDetected numeric (price) columns: %s\n", strjoin(priceCols, ", "));

P = pricesTT{:, isNumCol};        % T x Nassets price matrix
[T, N] = size(P);
fprintf("Price matrix: %d days x %d assets\n", T, N);

%% Log returns (consistent with the demo's price2ret 'continuous' mode)

R = diff(log(P));                  % (T-1) x N log returns
fprintf("Return matrix: %d days x %d assets\n", size(R,1), size(R,2));

%% Descriptive statistics per asset
% Assume daily data; annualize by sqrt(252).

dailyMu  = mean(R, 1);
dailyVol = std(R, 0, 1);
annMu    = dailyMu * 252;
annVol   = dailyVol * sqrt(252);
skewR    = skewness(R, 0, 1);
kurtR    = kurtosis(R, 0, 1);      % normal = 3, >3 means fat tails
sharpe   = annMu ./ annVol;

statsTbl = table(priceCols', dailyMu', dailyVol', annMu', annVol', ...
    sharpe', skewR', kurtR', ...
    'VariableNames', {'Asset','DailyMean','DailyStd', ...
        'AnnReturn','AnnVol','Sharpe','Skewness','Kurtosis'});
fprintf("\n=== Per-asset descriptive statistics ===\n");
disp(statsTbl)

%% Plot 1: Price trajectories (normalized to 1 at t=0)

figure('Name','Normalized Price Paths');
plot(P ./ P(1,:), 'LineWidth', 1.2);
legend(priceCols, 'Location','best');
xlabel('Day'); ylabel('Normalized Price');
title('Price Trajectories (normalized to 1)');
grid on;

%% Plot 2: Return distributions vs normal

figure('Name','Return Distributions');
for i = 1:N
    subplot(1, N, i);
    histogram(R(:,i), 60, 'Normalization','pdf'); hold on;
    xg = linspace(min(R(:,i)), max(R(:,i)), 200);
    plot(xg, normpdf(xg, dailyMu(i), dailyVol(i)), 'r', 'LineWidth', 1.5);
    title(sprintf('%s (kurt=%.2f)', priceCols{i}, kurtR(i)));
    xlabel('Log return'); ylabel('Density');
    grid on;
end
sgtitle('Daily Log Returns vs Normal Fit');

%% Plot 3: QQ plot — is it GBM?
% If the simulated data is from geometric Brownian motion, log returns
% should be Gaussian and QQ points fall on the diagonal.

figure('Name','QQ Plot vs Normal');
for i = 1:N
    subplot(1, N, i);
    qqplot(R(:,i));
    title(priceCols{i});
end
sgtitle('QQ Plot — deviations from line indicate non-Gaussianity');

%% Plot 4: Correlation matrix

corrR = corr(R);
figure('Name','Return Correlation');
imagesc(corrR); colormap(parula); colorbar; clim([-1 1]);
set(gca, 'XTick', 1:N, 'XTickLabel', priceCols, ...
         'YTick', 1:N, 'YTickLabel', priceCols);
title('Daily Log Return Correlation Matrix');
for i = 1:N
    for j = 1:N
        text(j, i, sprintf('%.2f', corrR(i,j)), ...
            'HorizontalAlignment','center', 'Color','k');
    end
end

%% Plot 5: Rolling 20-day volatility (annualized)

window = 20;
rollVol = zeros(size(R,1)-window+1, N);
for i = 1:N
    rollVol(:,i) = movstd(R(:,i), [window-1 0], 'Endpoints','discard') * sqrt(252);
end

figure('Name','Rolling Volatility');
plot(rollVol, 'LineWidth', 1.1);
legend(priceCols, 'Location','best');
xlabel('Day'); ylabel('Annualized Volatility');
title(sprintf('Rolling %d-day Volatility (annualized)', window));
grid on;

%% Plot 6: Autocorrelation of returns and squared returns
% Returns: should be near zero at all lags if GBM
% Squared returns: nonzero ACF suggests volatility clustering (GARCH-like)

maxLag = 20;
figure('Name','Autocorrelation');
for i = 1:N
    [acfR,  lagsR]  = localACF(R(:,i),    maxLag);
    [acfR2, lagsR2] = localACF(R(:,i).^2, maxLag);
    ci = 1.96 / sqrt(size(R,1));   % 95% CI under iid null

    subplot(2, N, i);
    stem(lagsR, acfR, 'filled'); hold on;
    yline( ci, '--r'); yline(-ci, '--r');
    title(sprintf('%s — return ACF', priceCols{i}));
    xlabel('Lag'); ylabel('ACF'); ylim([-0.3 1]); grid on;

    subplot(2, N, N + i);
    stem(lagsR2, acfR2, 'filled'); hold on;
    yline( ci, '--r'); yline(-ci, '--r');
    title(sprintf('%s — squared return ACF', priceCols{i}));
    xlabel('Lag'); ylabel('ACF'); ylim([-0.3 1]); grid on;
end

%% Local helper — manual ACF (no toolbox dependency)
function [acf, lags] = localACF(x, maxLag)
    x = x(:) - mean(x);
    n = numel(x);
    v0 = sum(x.^2);
    lags = (0:maxLag)';
    acf  = zeros(maxLag+1, 1);
    for k = 0:maxLag
        acf(k+1) = sum(x(1:end-k) .* x(k+1:end)) / v0;
    end
end

%% Summary checks — what kind of process is this?

fprintf("\n=== Summary diagnostics ===\n");
fprintf("Total trading days: %d\n", T);
fprintf("Implied years (assuming 252/yr): %.1f\n", T/252);
fprintf("Mean kurtosis across assets: %.2f  (normal = 3)\n", mean(kurtR));
fprintf("Mean abs skewness: %.2f  (normal = 0)\n", mean(abs(skewR)));
fprintf("Mean pairwise correlation: %.2f\n", ...
    mean(corrR(triu(true(N), 1))));

%% Implications for RL pipeline
%
% Things this script should answer (write findings in week3.md):
%   1. Is it daily, weekly, or monthly data? (Affects horizonPeriods semantics.)
%   2. How many years of history are available?
%   3. Is it GBM (Gaussian log returns) or something richer with fat tails /
%      vol clustering? If GBM, the demo's mean-variance frontier is exactly
%      optimal in-sample, leaving little for the RL agent to learn.
%   4. How correlated are the assets? Highly correlated → frontier collapses,
%      action discretization becomes less meaningful.
%   5. Are there regime shifts (visible in rolling vol plot)?
%
