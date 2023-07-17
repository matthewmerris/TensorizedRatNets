%% developing the loewnerization based pruning regiment
% load data for the first layer activations
act_in_path = "./tmp/in0.mat";
act_out_path = "./tmp/out1.mat";

act_in = double(cell2mat(struct2cell(load(act_in_path))));
act_out = double(cell2mat(struct2cell(load(act_out_path))));


%% capture dimensions for activation tensor and loewnerize (sub set of full set)
sz = size(act_in);
% mode3 = sz(1); % full size of mode 3
mode3 = 1000; % sub set of full set
mode1 = sz(2)/2;
if mod(sz(2),2) == 0
    mode2 = sz(2)/2;
else
    mode2 = sz(2)/2 + 1;
end

act_ten = zeros(mode1, mode2, mode3);
for i = 1:mode3
    temp_mat = loewnerize(act_in(i,:), 'T', act_out(i, :));
    act_ten(:,:,i) = temp_mat;
end

%%
L = 10.*ones(3,1);

tic; act_model = ll1(act_ten, L); toc;