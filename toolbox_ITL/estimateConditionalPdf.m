function [pxgivc, pxc, deltaX] = estimateConditionalPdf(D,x_i,w)

% D: cell array
% w = [kernel for X] is actually sigma


deltaX = x_i(2)-x_i(1);
numClass = length(D);
numEstimate = length(x_i);
pxc = zeros(numEstimate,numClass); % the joint pdf p(x,c)
for i = 1:numClass
    sample = D{1,i}; % N x D; N observations, each having D dim
    
    for j = 1:numEstimate
        mu = x_i(j,:);
        tmp = mvnpdf(sample, mu, w);
        pxc(j,i) = sum(tmp(:),1);
    end
end

% normalize within each column (class) of p(x,c)
pxc = pxc/sum(sum(pxc,1),2); % normalize the joint pxc

psum = sum(pxc,1);
pxgivc = pxc./repmat(psum,size(pxc,1),1);
if any(sum(pxgivc,1)-1 > 0.01) % test if the pdf p(x|c) sum to 1
    error('the p(x|c) is not normalized!!!');
end