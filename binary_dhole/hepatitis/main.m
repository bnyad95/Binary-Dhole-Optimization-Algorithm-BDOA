clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Specify the file name (ensure the file is in your working directory)
filename = 'hepatitis.data';

% Set import options
opts = delimitedTextImportOptions("NumVariables", 20);
opts.Delimiter = ",";
opts.VariableTypes = repmat("double", 1, 20);
opts.DataLines = [1, Inf];

% Read the data into a table
hepatitisTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(hepatitisTable));

% Extract features and labels
X = hepatitisTable{:, 2:20}; % Features (excluding the first column)
y = hepatitisTable{:, 1}; % Labels (first column)

% Display the size of the numerical data to ensure consistency
disp(size(X));
disp(length(y));

% Split the data into training and testing sets
HO = cvpartition(y, 'HoldOut', 0.2);
% XTrain = X(training(HO), :);
% yTrain = y(training(HO));
feat = X;
label = y;

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
title('Hepatitis'); grid on; hold on;