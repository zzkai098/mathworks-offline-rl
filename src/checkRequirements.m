function ok = checkRequirements()
%CHECKREQUIREMENTS  Verify the MATLAB release and toolboxes this project needs.
%   ok = checkRequirements()
%
%   Prints one row per dependency and returns true when every REQUIRED item is
%   both installed and licensed. Intended as the first thing a fresh clone runs:
%
%       matlab -batch "addpath('src'); checkRequirements"
%
%   Each toolbox below is listed with the project function that pulls it in, so
%   a failure points straight at the code that will break. Optimization Toolbox
%   is reported as OPTIONAL: the efficient frontier is built through Financial
%   Toolbox's Portfolio object, which uses its own built-in solver here (no
%   quadprog / setSolver call exists anywhere in src/), but MathWorks documents
%   Optimization Toolbox as a recommended companion for Portfolio workflows.
%
%   Returns ok = false rather than erroring, so callers can branch on it.

    MIN_RELEASE = 'R2025b';

    % name, license feature code, required?, what pulls it in
    deps = {
        'Reinforcement Learning Toolbox',      'RL_Toolbox',           true,  'rlDQNAgent, trainFromData'
        'Financial Toolbox',                   'Financial_Toolbox',    true,  'Portfolio, estimateFrontier'
        'Deep Learning Toolbox',               'Neural_Network_Toolbox', true, 'dlnetwork, dlfeval, adamupdate'
        'Statistics and Machine Learning Toolbox', 'Statistics_Toolbox', true, 'prctile, skewness, normcdf'
        'Optimization Toolbox',                'Optimization_Toolbox', false, 'not called directly; Portfolio companion'
    };

    fprintf('\nMATLAB release   %-12s (project developed on %s)\n', version('-release'), MIN_RELEASE);
    if ~strcmp(version('-release'), MIN_RELEASE)
        fprintf('    note: a different release may still work, but trainFromData\n');
        fprintf('          offline-RL support requires R2023b or newer.\n');
    end
    fprintf('\n%-42s %-10s %s\n', 'Toolbox', 'Status', 'Used by');
    fprintf('%s\n', repmat('-', 1, 96));

    ok = true;
    for k = 1:size(deps, 1)
        [name, feature, required, usedBy] = deps{k, :};

        installed = ~isempty(ver(toolboxDirName(feature)));
        licensed  = license('test', feature) == 1;

        if installed && licensed
            status = 'OK';
        elseif required
            status = 'MISSING';
            ok = false;
        else
            status = 'absent';
        end

        if ~required && ~strcmp(status, 'OK')
            status = 'optional';
        end

        fprintf('%-42s %-10s %s\n', name, status, usedBy);
    end

    fprintf('%s\n', repmat('-', 1, 96));
    if ok
        fprintf('All required toolboxes present.\n\n');
    else
        fprintf(2, 'Missing required toolbox(es) above — training/eval will fail.\n\n');
    end
end


function d = toolboxDirName(feature)
%TOOLBOXDIRNAME  Map a license feature code to the ver() directory name.
    switch feature
        case 'RL_Toolbox',              d = 'rl';
        case 'Financial_Toolbox',       d = 'finance';
        case 'Neural_Network_Toolbox',  d = 'nnet';
        case 'Statistics_Toolbox',      d = 'stats';
        case 'Optimization_Toolbox',    d = 'optim';
        otherwise,                      d = feature;
    end
end
