clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
filename = 'lung-cancer.data';

% Read the data into a table
opts = detectImportOptions(filename, 'FileType','text','Delimiter', ',');
opts.MissingRule = 'omitrow'; % Remove rows with missing data
lungCancerTable = readtable(filename, opts);

% Check the size of each column to identify inconsistencies
columnLengths = varfun(@length, lungCancerTable, 'OutputFormat', 'uniform');
% disp(columnLengths);

% Remove rows with missing data if necessary
lungCancerTable = rmmissing(lungCancerTable);

% Verify the table after removing missing data
% disp(head(lungCancerTable));

% Extract features (columns 2 to end) and labels (column 1)
X = table2array(lungCancerTable(:, 2:end));
y = lungCancerTable{:, 1};

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
title('Lung Cancer'); grid on; hold on;