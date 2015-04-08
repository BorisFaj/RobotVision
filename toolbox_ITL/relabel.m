function [B, label_org] = relabel(A)
%relabel rearranges matrix entries according to its rank order, so that the
%smallest number will start with rank 1, and so on
%
% ===== Example ====
% X = [-1.2000   -1.2000   20.3200    5.0000    5.0000
%      20.3200    5.0000    8.0000    5.0000    5.0000
%      20.3200    8.0000   20.3200    5.0000    8.0000
%      20.3200   20.3200   -1.2000    5.0000    8.0000
%       8.0000   20.3200    8.0000    8.0000    8.0000]
% 
% [B,map] = relabel(X);
% 
% B =
% 
%      1     1     4     2     2
%      4     2     3     2     2
%      4     3     4     2     3
%      4     4     1     2     3
%      3     4     3     3     3
% 
% map =
% 
%    -1.2000
%     5.0000
%     8.0000
%    20.3200     
%      
% ======= mapping =====
%    -1.2000 ---> 1
%     5.0000 ---> 2
%     8.0000 ---> 3
%    20.3200 ---> 4 

%% === code content ====

label_org = unique(A(:));
B = A;
for i = 1:length(label_org)
    B(A==label_org(i)) = i;
end

end

