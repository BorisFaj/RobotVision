function [D, classOut] = makeClassSampleCellArray(x,c)

% x: N x 1 vector, each representing a sample
% cIndex: N x 1 categorical vector ranging from 1 to N
% D: 1 x C cell array, where C denotes the number of classes  

c = relabel(c); % relabel so that the label ranges from 1 to N
c = c(:);
classList = unique(c');
numClass = length(classList);
classOut = zeros(1,numClass);
D = cell(1,numClass);
for i = 1:numClass
    D{1,i} = x(c==i,1);
    classOut(i) = i;
end

