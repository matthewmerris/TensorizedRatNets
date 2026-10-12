function [U,X] = test_blokcondia(test)
% test - boolean, if 1 produces exact random orthogonal LL1 tensor, if 0 produces minimal exact LL1 tensor

if test
    % test case parameters
    I = 20;
    J = 20;
    K = 10;
    R = 7;
    L = 4;
    % exact random orthogonal LL1 model
    U = cell(1,R);
    for r = 1:R
        % random orthonormal factors
        [Qa,~] = qr(randn(I,L),0);
        [Qb,~] = qr(randn(J,L),0);
        % random core
        Sr = randn(L,L);
        % component vector
        cr = zeros(K,1);
        cr(r) = 1;

        U{r} = {Qa, Qb, cr, Sr};
    end
    % build exact tensor
    X = zeros(I,J,K);
    for r = 1:R
        Ar = U{r}{1};
        Br = U{r}{2};
        cr = U{r}{3};
        Sr = U{r}{4};

        Mr = Ar * Sr * Br';
        for k = 1:K
            X(:,:,k) = X(:,:,k) + Mr * cr(k);
        end
    end
else
    % test case parameters
    I = 4;
    J = 4;
    K = 2;
    R = 2;
    L = 2;

    % build exact LL1 model
    U = cell(1,R);

    % Block 1
    A1 = [1 0;
          0 1;
          0 0;
          0 0];
    B1 = [1 0;
          0 1;
          0 0;
          0 0];
    c1 = [1;
          0];
    S1 = eye(L);
    U{1} = {A1, B1, c1, S1};

    % Block 2 
    A2 = [0 0;
          0 0;
          1 0;
          0 1];
    B2 = [0 0;
          0 0;
          1 0;
          0 1];
    c2 = [0;
          1];
    S2 = eye(L);
    U{2} = {A2, B2, c2, S2};

    % build exact tensor
    X = zeros(I,J,K);

    for r = 1:R
        Ar = U{r}{1};
        Br = U{r}{2};
        cr = U{r}{3};
        Sr = U{r}{4};

        Mr = Ar * Sr * Br';
        for k = 1:K
            X(:,:,k) = X(:,:,k) + Mr * cr(k);
        end
    end
end
