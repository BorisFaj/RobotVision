function div = KLDiv1(P,Q)
%Calculate KL divergence KL(P||Q)
% P and Q: discrete random variables of the same support
% which also means P and Q must, at least, be of the same dimension.
% deltaX: the step size when estimating the pdf P and Q
% if P and Q are intrinsically discrete distribution, deltaX = 1
% div: the divergence (scalar)

% Organize and Renormalize
P = P(:); P = P/sum(P(:));
Q = Q(:); Q = Q/sum(Q(:));
epsilon = 1e-14;

% eliminate "virtual/almost zero" and renormalize
Q(Q<epsilon) = 0; Q = Q/sum(Q(:),1);
% assumption: Qi=0 --> Pi=0
P(Q==0) = 0;
P(P<epsilon) = 0; 
if sum(P,1)<epsilon
    div = inf;
    return;
end


% === Get rid of i such that Pi=0
Q = Q(P~=0);
P = P(P~=0);
 
% Then we can calculate KL divergence safely
div = P'*(log(P)-log(Q));

end

