clear;

load('Obj1.mat');
load('Test1.mat');

for n = 1:(size(Obj1,2)-1)
    nombres(n) = cellstr(strcat('a',int2str(n)));
end
nombres(size(Obj1,2)) = cellstr('clase');

train = matlab2weka('obj1-train',nombres',Obj1);
test =  matlab2weka('obj1-test',nombres,Test1);

saveARFF('wekaInterface/Obj1.arff', train)
saveARFF('wekaInterface/Test1.arff', test)

load('Obj2.mat');
load('Test2.mat');

train = matlab2weka('obj2-train',nombres',Obj2);
test =  matlab2weka('obj2-test',nombres,Test2);

saveARFF('wekaInterface/Obj2.arff', train)
saveARFF('wekaInterface/Test2.arff', test)

load('Obj3.mat');
load('Test3.mat');

train = matlab2weka('obj3-train',nombres',Obj3);
test =  matlab2weka('obj3-test',nombres,Test3);

saveARFF('wekaInterface/Obj3.arff', train)
saveARFF('wekaInterface/Test3.arff', test)

load('Obj4.mat');
load('Test4.mat');

train = matlab2weka('obj4-train',nombres',Obj4);
test =  matlab2weka('obj4-test',nombres,Test4);

saveARFF('wekaInterface/Obj4.arff', train)
saveARFF('wekaInterface/Test4.arff', test)

load('Obj5.mat');
load('Test5.mat');

train = matlab2weka('obj5-train',nombres',Obj5);
test =  matlab2weka('obj5-test',nombres,Test5);

saveARFF('wekaInterface/Obj5.arff', train)
saveARFF('wekaInterface/Test5.arff', test)

load('Obj6.mat');
load('Test6.mat');

train = matlab2weka('obj6-train',nombres',Obj6);
test =  matlab2weka('obj6-test',nombres,Test6);

saveARFF('wekaInterface/Obj6.arff', train)
saveARFF('wekaInterface/Test6.arff', test)

load('Obj7.mat');
load('Test7.mat');

train = matlab2weka('obj7-train',nombres',Obj7);
test =  matlab2weka('obj7-test',nombres,Test7);

saveARFF('wekaInterface/Obj7.arff', train)
saveARFF('wekaInterface/Test7.arff', test)

load('Obj8.mat');
load('Test8.mat');

train = matlab2weka('obj8-train',nombres',Obj8);
test =  matlab2weka('obj8-test',nombres,Test8);

saveARFF('wekaInterface/Obj8.arff', train)
saveARFF('wekaInterface/Test8.arff', test)

clear;