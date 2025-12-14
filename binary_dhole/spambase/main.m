clc, clear, close;
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Specify the file name (ensure the file is in your working directory)
filename = 'spambase.data';

% Read the data into a table
opts = delimitedTextImportOptions("NumVariables", 58);
opts.Delimiter = ",";
opts.VariableTypes = repmat({'double'}, 1, 58);
opts = setvaropts(opts, 1:57, 'FillValue', NaN);

spambaseTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(spambaseTable));

% Extract features and labels
X = table2array(spambaseTable(:, 1:end-1));
y = table2array(spambaseTable(:, end));

% Display the size of X and y to ensure they are consistent
disp(size(X));
disp(length(y));

% Split the data into training and testing sets
HO = cvpartition(y, 'HoldOut', 0.2);

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
title('Spambase'); grid on; hold on;