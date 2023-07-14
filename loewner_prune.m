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
act_in = double(cell2mat(struct2cell(act_in)));

act_out = load(p.Results.act_out);
act_out = double(cell2mat(struct2cell(act_out)));

sz = size(act_in);
mode3 = sz(1);

mode1 = sz(2)/2;
if mod(sz(2),2) = 0:
    mode2 = mode1;
else
    mode2 = mode1 + 1;
end


%% preallocate and construct loewnerized tensor
act_ten = zeros(mode1, mode2, mode3);
for k = 1:mode3:
    loewn = loewnerize(act_out(k,:), 'T', act_in(k,:));
    act_ten(:,:,k) = loewn;
end

%% perform rank-(L,L,1) term decompostion

%% gather norms for each of the resulting terms

%% isolate terms with smallest norms

%% generate list of associated nodes (i.e. the crummy nodes)
  

prune_list = 0;
end

