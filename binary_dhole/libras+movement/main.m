clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Specify the file name (ensure the file is in your working directory)
filename = 'movement_libras.data';

% Read the data into a table
opts = delimitedTextImportOptions("NumVariables", 91);
opts.Delimiter = ",";
opts.VariableTypes = repmat("double", 1, 91);
opts.DataLines = [1, Inf];

movementLibrasTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(movementLibrasTable));

% Extract features and labels
features = movementLibrasTable{:, 1:90}; % First 90 columns are features
labels = movementLibrasTable{:, 91}; % 91st column is the label

% Display the size of the numerical data to ensure consistency
disp(size(features));
disp(length(labels));

% Split the data into training and testing sets
HO = cvpartition(labels, 'HoldOut', 0.2);

feat =features;
label = labels;

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
title('movement libras'); grid on; hold on;