function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%%%%%%%%%%%%%% Cost function without Regularization %%%%%%%%%%%%%%%%%%%%%

error = X*Theta' - Y;
squared_error = error.^2;

% To consider elements where r(i,j) = 1. Cancelling out elemenets for which r(i,j) = 0 by multiplying R .* sqaured_error
squared_error = squared_error .* R;

J = sum( sum (squared_error));
J = J / 2;

%****** Cost function with Regularization********%

if(lambda != 0)
	regularized_component_of_theta = Theta'.^2;
	regularized_component_of_theta = sum(sum(regularized_component_of_theta));

	regularized_component_of_X = X'.^2;
	regularized_component_of_X =  sum(sum(regularized_component_of_X));

	regularized_component = (regularized_component_of_theta + regularized_component_of_X) * lambda / 2 ;
	
	J = J + regularized_component;
end
%%%%%%%%%%%%%% Gradient without Regularization %%%%%%%%%%%%%%%%%%%%%


error = X*Theta' - Y;
%Cancelling out terms in ith row for which r(i,j) = 0
error = error .* R ;
X_grad = error * Theta; 



error = X*Theta' - Y;
error = error' ;
%Cancelling out terms in ith row for which r(i,j) = 0
error = error .* R';
Theta_grad = error * X; 


if(lambda != 0)
	X_grad = X_grad + X .* lambda;
end

if(lambda != 0)
	Theta_grad = Theta_grad + Theta .* lambda;
end





% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
