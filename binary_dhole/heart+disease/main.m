clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Specify the file name (ensure the file is in your working directory)
filename = 'processed.cleveland.data';

% Read the data into a table
opts = delimitedTextImportOptions("NumVariables", 14);
opts.Delimiter = ",";
opts.VariableNames = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts = setvaropts(opts, "ca", "TreatAsMissing", 'NaN', "EmptyFieldRule", "auto");
opts = setvaropts(opts, "thal", "TreatAsMissing", 'NaN', "EmptyFieldRule", "auto");
heartDataTable = readtable(filename, opts);

% Display the first few rows of the table to verify the data
disp(head(heartDataTable));

% Handle missing values (remove rows with NaN values)
heartDataTable = rmmissing(heartDataTable);

% Extract features and labels
X = table2array(heartDataTable(:, 1:end-1));
y = table2array(heartDataTable(:, end));

% Convert labels to binary (0: no heart disease, 1: heart disease)
y(y > 0) = 1;

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
    
    
figure (1);
plot(1:max_Iter,curvee,'LineWidth',2);
xlabel('Number of iterations');
ylabel('Fitness Value');
title('Heart Disease'); grid on; hold on;