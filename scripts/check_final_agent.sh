#!/usr/bin/env bash
# Genuine clean-session acceptance test: a FRESH MATLAB process loads FinalAgent.mat
# and runs predictAction — catches deps that live in a warm session but not a cold load.
set -uo pipefail
cd "$(dirname "$0")/.."
/Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath('src'); S=load('experiments/models/FinalAgent.mat'); FA=S.FA; [a,w]=predictAction(FA,101000,5,[4.2 -0.5 18 5.3]); fprintf('CLEAN-SESSION OK: action %d, weights sum %.3f, learner: %s\n', a, sum(w), FA.meta.learner);"
