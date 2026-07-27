%% Plot week9c convergence sweep: Q divergence + eval vs training length
T = readtable('experiments/logs/week9c_results.csv');
% dedup (seed,ME) keeping first (the ME=100/seed1000 repro row is identical)
[~, ia] = unique(T(:, {'seed','ME'}), 'rows', 'stable');
T = T(ia, :);
seeds = unique(T.seed);
cols = lines(numel(seeds));

f = figure('Visible','off','Position',[100 100 1200 450]);

subplot(1,2,1); hold on;
for i = 1:numel(seeds)
    s = T(T.seed==seeds(i), :);  s = sortrows(s, 'ME');
    semilogy(s.ME, s.qGlobal, '-o', 'LineWidth', 2, 'Color', cols(i,:), ...
        'DisplayName', sprintf('seed %d', seeds(i)));
end
set(gca,'YScale','log');
xlabel('training epochs (MaxEpochs)'); ylabel('val mean_s max_a Q(s,a)  [log]');
title('Q DIVERGES with training (offline deadly triad)');
legend('Location','northwest'); grid on;

subplot(1,2,2); hold on;
for i = 1:numel(seeds)
    s = T(T.seed==seeds(i), :);  s = sortrows(s, 'ME');
    plot(s.ME, s.evalSharpe, '-o', 'LineWidth', 2, 'Color', cols(i,:), ...
        'DisplayName', sprintf('seed %d', seeds(i)));
end
xlabel('training epochs (MaxEpochs)'); ylabel('hold-out eval Sharpe');
title('Eval never stabilizes (huge across-seed spread)');
legend('Location','best'); grid on;

exportgraphics(f, 'experiments/figures/week9c_divergence.png', 'Resolution', 150);
fprintf('saved -> experiments/figures/week9c_divergence.png\n');

% print a tidy summary table
fprintf('\nME  |   seed1000    seed2000    seed3000   (qGlobal)\n');
for me = [50 100 200 400]
    fprintf('%3d | ', me);
    for i = 1:numel(seeds)
        v = T.qGlobal(T.seed==seeds(i) & T.ME==me);
        if isempty(v); fprintf('%10s ', '-'); else; fprintf('%10.1f ', v(1)); end
    end
    fprintf('\n');
end
