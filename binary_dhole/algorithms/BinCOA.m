
function [best_fun,sFeat,Sf,Nf,global_Cov]  =BinCOA(feat,label,N,T,alpha,HO)
%% Define Parameters
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
fobj=@jFitnessFunction;
dim= size(feat,2);
cuve_f=zeros(1,N); 
% X=zeros(N,dim); %Initialize population
X=initialization(N,dim,ub,lb);
Sol=zeros(N,dim);
global_Cov = zeros(1,N);
Best_fitness = inf;
best_position = zeros(1,dim);
fitness_f = zeros(1,N);

% Initialize the population/solutions
% for i=1:N,
%     for j=1:dim % For dimension
%         if rand<=0.5
%             Sol(i,j)=0;
%         else
%             Sol(i,j)=1;
%         end
%     end
% end

for i=1:N
   fitness_f(i) =  fobj(feat,label,(X(i,:) > thres),HO); %Calculate the fitness value of the function
   
   if fitness_f(i)<Best_fitness
       Best_fitness = fitness_f(i);
       localBest_position = X(i,:);
   end
end
global_position = localBest_position; 
global_fitness = Best_fitness;
cuve_f(1)=Best_fitness;
t=1;
while(t<=T)
    C = 2-(t/T); %Eq.(7)
    temp = rand*15+20; %Eq.(3)
    xf = (best_position+global_position)/2; % Eq.(5)
    Xfood = best_position;
    for i = 1:N
        if temp>30
            %% summer resort stage
            if rand<0.5
                Xnew(i,:) = X(i,:)+C*rand(1,dim).*(xf-X(i,:)); %Eq.(6)
            else
            %% competition stage
                for j = 1:dim
                    z = round(rand*(N-1))+1;  %Eq.(9)
                    Xnew(i,j) = X(i,j)-X(z,j)+xf(j);  %Eq.(8)
                end
            end
        else
            %% foraging stage
            P = 3*rand*fitness_f(i)/fobj(feat,label,Xfood > thres,HO); %Eq.(4)
            if P>2   % The food is too big
                 Xfood = exp(-1/P).*Xfood;   %Eq.(12)
                for j = 1:dim
                    Xnew(i,j) = X(i,j)+cos(2*pi*rand)*Xfood(j)*p_obj(temp)-sin(2*pi*rand)*Xfood(j)*p_obj(temp); %Eq.(13)
                end
            else
                Xnew(i,:) = (X(i,:)-Xfood)*p_obj(temp)+p_obj(temp).*rand(1,dim).*X(i,:); %Eq.(14)
            end
        end
    end
    %% boundary conditions
    for i=1:N
        for j =1:dim
            if length(ub)==1
                Xnew(i,j) = min(ub,Xnew(i,j));
                Xnew(i,j) = max(lb,Xnew(i,j));
            else
                Xnew(i,j) = min(ub(j),Xnew(i,j));
                Xnew(i,j) = max(lb(j),Xnew(i,j));
            end
        end
    end
   %% Binary %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xnew(1,:);%%local
    for i = 1:N
        for j= 1:dim
            S_shaped_transfer= abs(1/(1+exp(-2*(Xnew(i,j))))); %% S-shape, 1th function
%             Bell_shaped_transfer= abs(exp(-(Xnew(i,j)/2)^2));
            if rand() > S_shaped_transfer
                Sol(i,j)=~Sol(i,j);
            else
                Sol(i,j)=Sol(i,j);
            end
        end
    end
    global_position = Sol(1,:);
    global_fitness = fobj(feat,label,global_position > thres,HO);
 
    for i =1:N
         %% Obtain the optimal solution for the updated population
        new_fitness = fobj(feat,label,(Sol(1,:)> thres),HO);
        if new_fitness<global_fitness
                 global_fitness = new_fitness;
                 global_position = Sol(1,:);
        end
        %% Update the population to a new location
        if new_fitness<fitness_f(i)
             fitness_f(i) = new_fitness;
             X(i,:) = Sol(1,:);
             if fitness_f(i)<Best_fitness
                 Best_fitness=fitness_f(i);
                 best_position = X(i,:);
             end
        end
    end
    global_Cov(t) = Best_fitness;
    t=t+1;
%     if mod(t,1)==0
%       disp("BinCOA"+"iter"+num2str(t)+": "+Best_fitness); 
%    end
end
 best_fun = Best_fitness;
 Pos   = 1:dim;
Sf    = Pos(( best_position > thres) == 1); 
sFeat = feat(:,Sf);
Nf    = length(Sf);
fprintf('BinCOA completed, ')
end
function y = p_obj(x)   %Eq.(4)
    y = 0.2*(1/(sqrt(2*pi)*3))*exp(-(x-25).^2/(2*3.^2));
end