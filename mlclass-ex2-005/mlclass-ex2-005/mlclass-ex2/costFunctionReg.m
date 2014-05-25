function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
