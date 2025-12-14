clc, close, clear;
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Load the CNAE-9 dataset
filename = 'CNAE-9.data'; % Update this with the correct filename

% Set import options
opts = delimitedTextImportOptions("NumVariables", 857);
opts.Delimiter = ",";
opts.VariableTypes = repmat("double", 1, 857);

% Read the data into a table
cnaeTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(cnaeTable));

% Split the data into features and labels
y = cnaeTable{:, 1};    % First column is the label
X = cnaeTable{:, 2:end}; % All other columns are features

% Normalize the features (if needed)
X = normalize(X);

% Display the size of the feature matrix and labels
disp(size(X));
disp(length(y));

% Split the data into training and testing sets
HO = cvpartition(y, 'HoldOut', 0.2);
% XTrain = X(training(cv), :);
% yTrain = y(training(cv));
% XTest = X(test(cv), :);
% yTest = y(test(cv));
%%%%%
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
title('CNAE-9'); grid on; hold on;