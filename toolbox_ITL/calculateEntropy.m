function Hx = calculateEntropy(px)
% Calculate entropy H(x) from a discrete distribution P(x) using log base e
% (ln)
% ===== INPUT ====
% px: discrete distribution P(x)--can be either row or column vector
% ===== OUTPUT =====
% Hx : entropy H(X) -- a scalar value

%% Example1: 
% px = [0.5 0.5];
% Hx = calculateEntropy(px);
% >> Hx = 0.6931 % (i.e., log(2))

%% content

px = px(:);
px = px/sum(px); % renormalize
epsilon = 1e-14;

%% % === approach I ===
% % Using for-loop and use if to get rid of px_i < epsilon
% % This approach can be slow, but easier to understand
% numX = length(px);
% hx = px*0;
% for i = 1:numX
%     if px(i)<epsilon
%         hx(i) = 0;
%     else
%         hx(i) = -px(i)*log(px(i));
%     end
% end
% Hx = sum(hx(:));
% % =================

%% === approach II ===
% Filter out the px_i < epsilon in the first place
% and the vectorize the operation
px = px(px>=epsilon);
px = px/sum(px); % renormalize
logpx = log(px);
Hx = -px'*logpx;
% ===================



