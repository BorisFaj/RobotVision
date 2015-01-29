function evaluateResults (results, clasesForTest,objectsForTest)

numWellClassified = 0;
numBadClassified = 0;
numNotClassified = 0;


numObjectDetected = zeros(numel(objectsForTest(:,1)),1);
numObjectNotDetected = zeros(numel(objectsForTest(:,1)),1);
numObjectBadDetected = zeros(numel(objectsForTest(:,1)),1);

for i=1:numel(clasesForTest)
    if(results(i,1)==0)
        numNotClassified = numNotClassified+1;
    else if(results(i,1)==clasesForTest(i))
            numWellClassified = numWellClassified +1;
        else
            numBadClassified = numBadClassified +1;
        end
    end
end

scoreRooms = numWellClassified-0.5*numBadClassified;

fprintf('RESULTS: WELL CLASSIFIED %d, BAD CLASSIFIED %d. NOT CLASSIFIED %d. \n',numWellClassified,numBadClassified,numNotClassified);
fprintf('\n');
fprintf('\n');
fprintf('### SCORE FROM ROOMS %4.2f ###',scoreRooms);
fprintf('\n');
fprintf('\n');


for i=1:numel(objectsForTest(1,:))
    for j=1:numel(objectsForTest(:,1))
        
        if(results(i,j)==0)
            if(objectsForTest(j,i)==1)
                numObjectNotDetected(j)=numObjectNotDetected(j)+1;  % FALSE NEGATIVE
            else
                % NOTHING --> TRUE NEGATIVE
            end
            
        else if(objectsForTest(j,i)==1)
                numObjectDetected(j)=numObjectDetected(j)+1;    % TRUE POSITIVE
            else
                numObjectBadDetected(j)=numObjectBadDetected(j)+1;  % FALSE POSITIVE
            end
        end
        
    end
end

for j=1:numel(objectsForTest(:,1))
    fprintf('OBJECT %d\n',j);
    fprintf('DETECTIONS %d, BAD DETECTIONS %d. MISSING DETECTIONS %d. \n',numObjectDetected(j),numObjectBadDetected(j),numObjectNotDetected(j));
    fprintf('### SCORE FROM %d OBJECT DETECTION: %4.2f%. ###',j,numObjectDetected(j)-0.5*numObjectBadDetected(j));
    fprintf('\n');
end

scoreObjects = sum(numObjectDetected(:))-0.25*sum(numObjectBadDetected(:))-0.25*sum(numObjectNotDetected(:));
fprintf('\n');
fprintf('### SCORE FROM ROOMS %4.2f ###',scoreRooms);
fprintf('\n');
fprintf('### SCORE FROM OBJECTS %4.2f ###',scoreObjects);
fprintf('\n');
fprintf('### FINAL SCORE  %4.2f ###',scoreRooms+scoreObjects);
fprintf('\n');
fprintf('### MAXIMUM SCORE  %4.2f (%4.2f FROM ROOMS AND %4.2f FROM OBJECTS). ###',numel(objectsForTest(1,:)) + sum(sum(objectsForTest)),numel(objectsForTest(1,:)),sum(sum(objectsForTest)));
fprintf('\n');