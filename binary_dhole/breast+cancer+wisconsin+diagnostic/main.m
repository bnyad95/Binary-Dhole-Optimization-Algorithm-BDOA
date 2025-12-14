clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Load the WDBC dataset
filename = 'wdbc.data';
formatSpec = '%d%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
fileID = fopen(filename, 'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', ',', 'HeaderLines', 0);
fclose(fileID);

% Convert to table
wdbcTable = table(dataArray{1}, dataArray{2}, dataArray{3:end}, 'VariableNames', ...
    {'ID', 'Diagnosis', 'RadiusMean', 'TextureMean', 'PerimeterMean', 'AreaMean', ...
    'SmoothnessMean', 'CompactnessMean', 'ConcavityMean', 'ConcavePointsMean', ...
    'SymmetryMean', 'FractalDimensionMean', 'RadiusSE', 'TextureSE', 'PerimeterSE', ...
    'AreaSE', 'SmoothnessSE', 'CompactnessSE', 'ConcavitySE', 'ConcavePointsSE', ...
    'SymmetrySE', 'FractalDimensionSE', 'RadiusWorst', 'TextureWorst', 'PerimeterWorst', ...
    'AreaWorst', 'SmoothnessWorst', 'CompactnessWorst', 'ConcavityWorst', 'ConcavePointsWorst', ...
    'SymmetryWorst', 'FractalDimensionWorst'});

% Convert diagnosis to binary values (M=1, B=0)
wdbcTable.Diagnosis = double(strcmp(wdbcTable.Diagnosis, 'M'));

% Extract features and labels
X = table2array(wdbcTable(:, 3:end));
y = wdbcTable.Diagnosis;

disp(size(X));
disp(length(y));
% Split the data into training and testing sets
HO = cvpartition(y, 'HoldOut', 0.2);
% XTrain = X(training(cv), :);
% yTrain = y(training(cv));
% XTest = X(test(cv), :);
% yTest = y(test(cv));
%%%%%
feat = table2array(wdbcTable(:, 3:end));
label = wdbcTable.Diagnosis;

% Parameter setting
N        = 10;
max_Iter = 30; 
alpha    = 2;

[BDOA_fitness,sFeat,Sf,Nfa,curvea] = Bell_BDOA(feat,label,N,max_Iter,alpha,HO);
BDO_Accuracy= jwrapperKNN(sFeat,label,HO)
    
    

% Plot convergence curve
figure (1);
plot(1:max_Iter,curvea,'LineWidth',2);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('WDBC'); grid on; hold on;
