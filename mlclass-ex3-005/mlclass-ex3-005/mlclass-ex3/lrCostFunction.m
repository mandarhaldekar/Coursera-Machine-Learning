function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%Logistic regression cost function
h = sigmoid(X * theta);
J = y' * log(h) + (1-y') * log(1-h);

J = -J/m;

%Regularizing it
temp = theta;
temp(1) = 0; % as we do not regularize for j = 0
J = J + lambda / (2*m) * sum(temp.^2);


% Calculating the gradient
h = sigmoid(X * theta);
error_term = h - y;  % (h(x)- y)

grad = error_term' * X;   % ( h(x) - y) * X . Here size(grad) 1 X (n+1)
grad = grad(:);   % Converting to column vector (n+1) X 1
grad = grad./m ;

%Regularizing gradient
grad = grad + lambda/m * temp;








% =============================================================



end
