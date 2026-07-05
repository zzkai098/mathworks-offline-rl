function [R, M_z] = inputTestData(priceFile, macroFile, macroMean, macroStd)
%INPUTTESTDATA  Load test prices/macro and z-score the macro block.
%   [R, M_z] = inputTestData(priceFile, macroFile, macroMean, macroStd)
%
%   Shared helper lifted verbatim from week6_1_regime_gated.m /
%   week7_continuous_action_ddpg.m so the backtest harness reconstructs the
%   test inputs exactly as the training scripts did.
%
%   R    : log returns, diff(log(P)) over the price columns (T-1 x Nassets)
%   M_z  : macro factors z-scored with the TRAIN-fitted macroMean/macroStd
%          (no look-ahead — pass the same params the agent trained under)

    pricesTT = readtable(priceFile, 'VariableNamingRule', 'preserve');
    P = pricesTT{:, 2:end};
    R = diff(log(P));

    macroTT = readtable(macroFile, 'VariableNamingRule', 'preserve');
    M = macroTT{:, 2:end};
    M_z = (M - macroMean) ./ macroStd;
end
