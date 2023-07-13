function prune_list = loewner_prune(varargin)
%LOEWNER_PRUNE Loewnerization of network activations for pruning NNs
%   Detailed explanation goes here
%   + Constructs a tensor by stacking loewner matrices in the third mode
%   + Loewner matrices are form using the matrices passed as filepaths in the
%   arguments
%   + (L,L,1) decomposition performed
%   + the norm of the subsequent terms is used as means of identifying
%   unimportant nodes for the layer in question
%% Handle input arguments and load data
p = inputParser;
addRequired(p,'act_in', @ischar);
addRequired(p,'act_out', @ischar);

if ~isfile(p.Results.act_in) || ~isfile(p.Results.act_out)
    warningMessage = sprintf('Warning: file(s) does not exist:\n%s | %s', ...
                            p.Results.act_in,p.Results.act_out);
    uiwait(msgbox(warningMessage));
end

act_in = load(p.Results.act_in);
act_in = struct2cell(act_in);

act_out = load(p.Results.act_out);
act_out = struct2cell(act_out);

  

prune_list = 0;
end

