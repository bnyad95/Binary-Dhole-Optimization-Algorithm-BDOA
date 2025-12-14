
% Notation: This fitness function is for demonstration 
function cost = jFitnessFunction(feat,label,X,HO)
% if sum(X == 1) == 0
%   cost = 1;
% else
%   cost = jwrapperKNN(feat(:, X == 1),label,HO);
% end
if sum(X == 1) == 0
    cost = 1;
else
    alpha=0.99;
    y   = jwrapperKNN(feat(:, X == 1),label,HO);
    beta= 1-alpha;
    Sf  = sum(X == 1);
    N   = size(feat,2);
    cost = alpha*y+(beta*(Sf/N));
end
end
function error = jwrapperKNN(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k = 5; 
xtrain = sFeat(HO.training == 1,:);
ytrain = label(HO.training == 1); 
xvalid = sFeat(HO.test == 1,:); 
yvalid = label(HO.test == 1); 
Model     = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
pred      = predict(Model,xvalid);
num_valid = length(yvalid); 
correct   = 0;
for i = 1:num_valid
  if isequal(yvalid(i),pred(i))
    correct = correct + 1;
  end
end
Acc   = correct / num_valid; 
error = 1 - Acc;
end
