function [MI, IGR] = mi_discrete(pxc)
% Calculate mutual information from the joint distribution p_{X,C}(x,c)
% ===== INPUT ====
% pxc: the discrete joint probability distribution P(x,c)
%   X : a continuous-value rv representing some measurement
%   C : a discrete-valued rv representing class-label/category 
% ===== OUTPUT =====
% MI : mutual information between X and C -- a scalar value
% IGR: Information Gain Ratio between X and C, essentially I(X;C)/H(X)
%
%% Example
% 
% pxc = [0 0 1;
%        0 0 1;
%        0 1 0;
%        0 1 0;
%        1 0 0;
%        1 0 0];
% [MI, IGR] = calculateMutualInformation(pxc);
% MI =
% 
%     1.0986
% 
% 
% IGR =
% 
%     0.6131

%% content
[numX, numC]= size(pxc);
epsilon = 1e-14;

pxc(pxc<epsilon) = 0; % get rid of the too-small value
pxc = pxc/sum(pxc(:)); % renormalize

% calculate the marginal (automatically normalized)
px = sum(pxc,2);
pc = sum(pxc,1); pc = pc(:)';

%% % ====== Approach 1 ======
% % Using for-loop, pretty slow
% mi = zeros(numX, numC);
% for c = 1:numC
%     for i = 1:numX
%         if pxc(i,c) < epsilon
%             mi(i,c) = 0;
%         else
%             mi(i,c) = pxc(i,c)*(log(pxc(i,c))-log(px(i)*pc(c)));
%         end
%     end
% end
% 
% MI = sum(sum(mi,1),2);
% % ==========================

%% % ====== Approach 2 ======
% This is a faster implementation by filtering out pxc=0, then using vectorization
px_hat = repmat(px,[1,numC]);
pc_hat = repmat(pc,[numX,1]);

px_hat = px_hat(pxc(:)~=0);
pc_hat = pc_hat(pxc(:)~=0);
pxc_hat = pxc(pxc(:)~=0);

MI = pxc_hat'*(log(pxc_hat)-log(px_hat)-log(pc_hat));
% ===========================

%% ====== Calculate Information Gain Ratio (IGR) =========
% IGR = I(C,X)/H(X)
Hx = entropy_discrete(px);
IGR = MI/Hx;
