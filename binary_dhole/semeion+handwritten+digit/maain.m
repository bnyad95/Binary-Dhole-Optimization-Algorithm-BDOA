clc, clear,close;
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Load the Semeion Handwritten Digit dataset
filename = 'semeion.data'; % Update this with the correct filename

% Set import options
opts = delimitedTextImportOptions("NumVariables", 266);
opts.Delimiter = " ";
opts.VariableTypes = repmat("double", 1, 266);
opts.ExtraColumnsRule = 'ignore';  % Ignore any extra columns
opts.EmptyLineRule = 'read';  % Handle empty lines
opts.ConsecutiveDelimitersRule = 'join'; % Handle multiple spaces

% Read the data into a table
semeionTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(semeionTable));
% Split the data into features and labels
X = semeionTable{:, 1:256}; % First 256 columns are features
y = semeionTable{:, 257:end}; % Last 10 columns are labels

% Convert the binary labels to a single digit label
[~, y] = max(y, [], 2); % Convert binary label matrix to a single column

% Display the size of the feature matrix and labels
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
title('semeion'); grid on; hold on;