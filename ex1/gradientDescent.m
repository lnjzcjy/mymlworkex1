function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
 good0 = 0;
 good1 = 0 ;
 for n0_forXsample = 1 : m
      good0= good0+(1/m) * (theta(1,1) + theta(2,1)*X(n0_forXsample,2) - y(n0_forXsample,1));
 end
 for n1_forXsample = 1 : m
      good1= good1+(1/m) * (theta(1,1) + theta(2,1)*X(n1_forXsample,2) - y(n1_forXsample,1)) * X(n1_forXsample,2);
 end
 
 theta(1,1) = theta(1,1) - alpha *  good0 ; 
 theta(2,1) = theta(2,1) - alpha *  good1 ;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

figure ; 
plot(J_history);
ylabel('j HISTORY');

end
