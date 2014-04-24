function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = ( X * theta - y ) .^ 2;
J = sum(J);
J = J / ( 2 * m);

%Regularization
regularization_term = 0 ; 
%Deleting theta(0)
theta_without_theta_zero = theta;
theta_without_theta_zero(1,:) = [];  %First row deleted

regularization_term = sum(theta_without_theta_zero .^ 2);
regularization_term = regularization_term * lambda / ( 2 * m);

%Final cost 
J = J + regularization_term;

% Gradient Descent

%First find gradient descent without regularization

temp =  X' * ( X * theta - y);
%row-wise sum
grad = sum(temp,2);
grad = grad / m;


% Regularization
regularization_term = zeros(size(grad));
regularization_term = theta .* (lambda/m);
regularization_term(1,:) = 0; % as no regularization for theta0

grad = grad + regularization_term;








% =========================================================================

grad = grad(:);

end
