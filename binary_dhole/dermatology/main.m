clc, clear, close;
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Load the Dermatology dataset
filename = 'dermatology.data'; % Update this with the correct filename

% Set import options
opts = delimitedTextImportOptions("NumVariables", 35);
opts.Delimiter = ",";
opts.VariableTypes = repmat("double", 1, 35); % Assuming all columns are numeric
opts.DataLines = [2, Inf]; % Assuming the first row contains headers

% Read the data into a table
dermatologyTable = readtable(filename, opts);

% Convert the class labels to categorical values
X = dermatologyTable{:, 1:end-1}; % All other columns are features
y = categorical(dermatologyTable{:, end}); % Last column is the label

% Normalize the features (if needed)
X = normalize(X);

% Display the size of the feature matrix and labels
disp(size(X));
disp(length(y));

% Split the data into training and testing sets
HO = cvpartition(y, 'HoldOut', 0.2);
XTrain = X(training(HO), :);
yTrain = y(training(HO));
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
title('Dertmatology'); grid on; hold on;