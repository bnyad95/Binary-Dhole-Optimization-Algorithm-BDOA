clc, clear, close;
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
load fisheriris
label=species;
feat=meas;

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho);
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
title('IRIS'); grid on; hold on;