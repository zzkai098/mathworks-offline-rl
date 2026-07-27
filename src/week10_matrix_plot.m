%% Final matrix: tuned-DQN (Polyak+LR1e-4) vs IQL — boundedness + across-seed eval-Sharpe spread
OUT = 'experiments/logs/matrix';
seeds = [1000 2000 3000]; MEs = [50 100 200 400];
qD=nan(4,3); shD=nan(4,3); qI=nan(4,3); shI=nan(4,3);
for i=1:4, for j=1:3
    fd = sprintf('%s/dqn_s%d_me%d.csv',OUT,seeds(j),MEs(i));
    if isfile(fd), r=readmatrix(fd); qD(i,j)=r(5); shD(i,j)=r(12); end
    fi = sprintf('%s/iql_s%d_me%d.csv',OUT,seeds(j),MEs(i));
    if isfile(fi), r=readmatrix(fi); qI(i,j)=r(6); shI(i,j)=r(13); end
end, end
ceil_ = 1.14;

f=figure('Visible','off','Position',[100 100 1300 430]);

% Panel 1: Q vs ME, both bounded
subplot(1,3,1); hold on;
for j=1:3
    plot(MEs,qD(:,j),'--o','Color',[0.85 0.3 0.1],'LineWidth',1.3);
    plot(MEs,qI(:,j),'-o','Color',[0.1 0.5 0.85],'LineWidth',1.7);
end
yline(ceil_,':k','economic ceiling','LineWidth',1.1);
xlabel('epochs'); ylabel('val mean max_a Q'); title('Both BOUNDED (DQN dashed / IQL solid)');
grid on; ylim([0 2]);

% Panel 2: across-seed eval-Sharpe spread (std) vs ME — the deciding metric
subplot(1,3,2); hold on;
plot(MEs, std(shD,0,2), '--s','Color',[0.85 0.3 0.1],'LineWidth',2,'DisplayName','tuned-DQN');
plot(MEs, std(shI,0,2), '-s','Color',[0.1 0.5 0.85],'LineWidth',2,'DisplayName','IQL');
xlabel('epochs'); ylabel('across-seed eval-Sharpe std'); title('Deciding metric: policy reproducibility');
legend('Location','best'); grid on;

% Panel 3: mean eval Sharpe +/- across-seed std
subplot(1,3,3); hold on;
errorbar(MEs, mean(shD,2), std(shD,0,2), '--s','Color',[0.85 0.3 0.1],'LineWidth',1.5,'DisplayName','tuned-DQN');
errorbar(MEs, mean(shI,2), std(shI,0,2), '-s','Color',[0.1 0.5 0.85],'LineWidth',1.5,'DisplayName','IQL');
xlabel('epochs'); ylabel('mean eval Sharpe (±seed std)'); title('Mean eval performance (overlapping)');
legend('Location','best'); grid on;

exportgraphics(f,'experiments/figures/week10_dqn_vs_iql.png','Resolution',150);
fprintf('saved -> experiments/figures/week10_dqn_vs_iql.png\n\n');

fprintf('ME  | DQN: Qmax  Sharpe(mean±std)  | IQL: Qmax  Sharpe(mean±std)\n');
for i=1:4
    fprintf('%3d | %8.2f  %.2f±%.2f          | %8.3f  %.2f±%.2f\n', MEs(i), ...
        max(qD(i,:)), mean(shD(i,:)), std(shD(i,:)), ...
        max(qI(i,:)), mean(shI(i,:)), std(shI(i,:)));
end
fprintf('\nMean across-seed Sharpe std (100-400): DQN %.3f | IQL %.3f\n', ...
    mean(std(shD(2:4,:),0,2)), mean(std(shI(2:4,:),0,2)));
