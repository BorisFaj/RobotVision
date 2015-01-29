function showDatasetStats( Configuration)


fprintf('\n');
fprintf('### TRAINING ###');
fprintf('\n');
fprintf('\n');
fprintf('### Rooms ###');
fprintf('\n');
fprintf('\n');
for i=1:Configuration.numClasses
    fprintf('Number of frames belonging to class %s: %d.\n',Configuration.labelClasses{i},numel(find(Configuration.trainingInfo.class==i)));
end

fprintf('\n');
fprintf('### Objects ###');
fprintf('\n');
    
for i=1:Configuration.numObjects
    fprintf('Number of frames with appearances of the object %s: %d.\n',Configuration.labelObjects{i},numel(find(Configuration.trainingInfo.objectAppearance(i,:)==1)));
end

fprintf('\n');
fprintf('\n');

fprintf('\n');
fprintf('### TEST ###');
fprintf('\n');
fprintf('\n');
fprintf('### Rooms ###');
fprintf('\n');
fprintf('\n');
for i=1:Configuration.numClasses
    fprintf('Number of frames belonging to class %s: %d.\n',Configuration.labelClasses{i},numel(find(Configuration.testInfo.class==i)));
end

fprintf('\n');
fprintf('### Objects ###');
fprintf('\n');
    
for i=1:Configuration.numObjects
    fprintf('Number of frames with appearances of the object %s: %d.\n',Configuration.labelObjects{i},numel(find(Configuration.testInfo.objectAppearance(i,:)==1)));
end

fprintf('\n');
fprintf('\n');

end

