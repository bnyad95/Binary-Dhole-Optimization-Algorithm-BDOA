%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )
% N        : Number of solutions
% max_Iter : Maximum number of iterations
% alpha    : Constant 
%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------
%% BDOA Algorithm
clc, clear, close; 
addpath('D:\PhD\DO algorithm\binary DOA\B_DOA\feature_selection\algorithms');
% Benchmark data set 
load ionosphere.mat; 
label=Y;
feat=X;
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
title('ionosphere'); grid on; hold on;