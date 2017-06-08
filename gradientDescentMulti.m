function [th, J_hist] = gradientDescentMulti(X, y, theta, alpha, num_iters)
 m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
     theta = theta -((1/m) * ((X * theta) - y)' * X)' * alpha;
end
th=theta;
J_hist=J_history;
end;    
 
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
   
