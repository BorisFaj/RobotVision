% test the ITL routine
clear; 
clc;
close all;

D1 = [0.2 0.5 0.3]; % H(D1) = 1.48548 bits (base2)
D2 = [0.3 0.1 0.6]; % H(D2) = 1.29546 bits (base2)

X1 = randn(10000,1);
X2 = X1+5;
X3 = sqrt(2)*X1+5;
%% Test discrete entropy
H1 = entropy_discrete(D1)/log(2) % convert to base2
H2 = entropy_discrete(D2)/log(2) % convert to base2
% So, the discrete entropy works fine here!
%% Test Entropy H(x)
HX1 = entropy_sample(X1,0.05,100)
HX2 = entropy_sample(X2,0.05,100)
% theoretical value = 1/2*log(2*e*pi*sigma^2) = 1.4189
% The calculated entropy here is not the same as the theoretical because we
% make it discrete.
%% Test KL divergence discrete distribution
D1 = [0 0.1 0.2 0.3 0.4 0   0   0   0   0];
D2 = [0 0   0.4 0.3 0.2 0.1 0   0   0   0];
D3 = [0 0   0   0.4 0.3 0.2 0.1 0   0   0];
D4 = [0 0   0   0   0.4 0.3 0.2 0.1 0   0];
D5 = [0 0   0   0   0   0.4 0.3 0.2 0.1 0];
KL11 = kl_discrete(D1,D1); % = 0
KL12 = kl_discrete(D1,D2); % = 0.1386
KL13 = kl_discrete(D1,D3); % = 0.0288
KL14 = kl_discrete(D1,D4); % = 0
KL15 = kl_discrete(D1,D5); % = inf
%% Test KL divergence
KL12 = kl_sample(X1,X2)
KL13 = kl_sample(X1,X3)
KL21 = kl_sample(X2,X1)
KL11 = kl_sample(X1,X1)
% expect: KL12>KL13, KL12 != KL21
% This is correct!

%% Test Jensen-Shannon divergence
JS12 = js_sample(X1,X2)
JS21 = js_sample(X2,X1)
JS13 = js_sample(X1,X3)
JS11 = js_sample(X1,X1)
JS22 = js_sample(X2,X2)
% Expect: JS12 = JS21, JS12>JS13
% This is correct