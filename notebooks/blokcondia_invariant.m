function [score, diagnostics] = blokcondia_invariant(X,U)
% X : tensor (I x J x K) - double or tensorlab tensor
% U : btd tensor - 1 x R cell where U{r} = {Ar, Br, cr, Sr}

% isolate number of terms
R = numel(U);

% isolate multi-linear rank of terms
L = size(U{1}{1}, 2);

% construct global factors
A = [];
B = [];
C = [];

block_CNs = zeros(R,1);

for r = 1:R
    Ar = U{r}{1};
    Br = U{r}{2};
    Cr = U{r}{3};
    Sr = U{r}{4};

    % absorb block core
    Mr = Ar * Sr * Br';

    % SVD canonicalization to remove projective ambiguity
    [Ur, Sr_svd, Vr] = svd(Mr, 'econ');
    Ar_canon = Ur(:,1:L);
    Br_canon = Vr(:,1:L) * Sr_svd(1:L,1:L);

    % normalize to stabilize pseudoinverse
    scale = sqrt(sum(Ar.^2, 1) + eps);
    Ar_canon = Ar_canon ./ scale;
    Br_canon = Br_canon .* scale;

    % accumulate
    A = [A, Ar_canon];
    B = [B, Br_canon];
    C = [C, Cr];

    block_CNs(r) = cond(Sr_svd(1:L,1:L));
end

% compute pseudoinverses for computing tucker core
Ap = pinv(A);
Bp = pinv(B);
Cp = pinv(C);

% compute tucker-like core
G = tmprod(X, {Ap, Bp, Cp}, [1 2 3]);
G = double(G);

% construct ideal block core
G_ideal = zeros(size(G));

for r = 1:R
    idx = (r-1)*L+1 : r*L;
    G_ideal(idx, idx, r) = G(idx, idx, r);
end

% compute score
off_block_energy = norm(G(:) - G_ideal(:))^2;
total_energy = norm(G(:))^2 + eps;

score = 100 * (1 - off_block_energy / total_energy);

% collect diagnostics
diagnostics = struct();

diagnostics.total_core_energy = total_energy;
diagnostics.off_block_energy = off_block_energy;
diagnostics.block_core_energy = norm(G_ideal(:))^2;
diagnostics.relative_offblock_energy = off_block_energy / total_energy;
diagnostics.block_condition_numbers = block_CNs;
diagnostics.mean_condition_number = mean(block_CNs);

% compute subspace overlap diagnostics
overlap_A = zeros(R);
overlap_B = zeros(R);

for i = 1:R
    idx_i = (i-1)*L +1 : i*L;
    Ai = A(:,idx_i);
    Bi = B(:,idx_i);

    for j = i+1:R
        idx_j = (j-1)*L + 1 : j*L;
        Aj = A(:, idx_j);
        Bj = B(:, idx_j);

        overlap_A(i,j) = norm(Ai' * Aj, 'fro')^2;
        overlap_B(i,j) = norm(Bi' * Bj, 'fro')^2;
    end
end
diagnostics.subspace_overlap_A = overlap_A;
diagnostics.subspace_overlap_B = overlap_B;

end
