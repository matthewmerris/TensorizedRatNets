function score = blokcondia(X,U)
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

for r = 1:R
    Ar = U{r}{1};
    Br = U{r}{2};
    Cr = U{r}{3};
    Sr = U{r}{4};

    % absorb Sr into Ar
    Ar = Ar * Sr;

    % normalize to stabilize pseudoinverse
    scale = sqrt(sum(Ar.^2, 1) + eps);
    Ar = Ar ./ scale;
    Br = Br .* scale;

    % accumulate
    A = [A, Ar];
    B = [B, Br];
    C = [C, Cr];
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

end
