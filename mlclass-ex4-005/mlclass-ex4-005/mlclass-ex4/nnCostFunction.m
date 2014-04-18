function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));



				 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%size(Theta2) = 10 x 26

% Setup some useful variables
m = size(X, 1);

%Add columns of 1's matrix
X = [ones(m,1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = X;
z2 = Theta1 * a1';
a2 = sigmoid(z2);
%Add column of ones to X (bias unit)
a2 = a2';
a2 = [ones(size(a2)(1),1) a2];
z3 = Theta2 * a2';
a3 = sigmoid(z3);
a3 = a3';
h = a3; %h is m x k dimensional matrix. h(i,k) for ith training example cost corresponding to class k

y_10 = zeros(max(y),1);
sum = 0;
for i = 1 : m,
	
	y_10 = zeros(max(y),1);
	temp = y(i);
	y_10(temp) = 1;
	
	sum = sum + y_10' * log(h(i,:))' + (1-y_10)' * log( 1 - h(i,:))';

end
	
J = -(sum)/m;




%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

y_10 = zeros(max(y),1);
Delta2 = zeros(size(Theta2_grad));
Delta1 = zeros(size(Theta1_grad));
z2 = z2';
z2 = [ones(size(z2)(1),1) z2]; % size of z2 = 5000 x 26

%Step 1 : All activations a1, a2, a3 are computed as above
for t = 1 : m,
	
	% Size of a3 = 5000 x 10
	
	
	% Output vector for this training example
	y_10 = zeros(max(y),1);
	temp = y(t);
	y_10(temp) = 1;
	
	% Step 2 :  Calculating delta term for the output
	delta_3 = a3(t,:)' - y_10;  % Size of delta_3 = 10 x 1
		
	%Step 3: for hidden layer i.e  l = 2
	
	delta_2 = Theta2' * delta_3 .* (sigmoidGradient(z2(t,:)))';
	
	%Removing delta_2(0)
	delta_2  = delta_2( 2 : end );  % size of delta_2 = 25 x 1
	
	
	% Step 4 : Accumulate in DELTA. Note Delta2 corresponds to Theta2_grad , Delta1 corresponds to Theta1_grad
	
	%Size of a2 = 5000 x 26
	Delta2 = Delta2 + delta_3 * a2(t,:);  % Size of Delta2 = 10 X 26
	
	%Size of a2 = 5000 x 401
	Delta1 = Delta1 + delta_2 * a1(t,:); % Size of Delta2 = 25 X 401
	

	
end

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 /m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Regularizing cost function. As only 3 layers. So, regularization from layer = 1 to layer = L-1 i.e 2
% l = 1 . Input layer
if lambda != 0,

Theta1_without_bias = Theta1;
Theta1_without_bias(:,1) = []; % Deleting first column . Size = 25 x 400

Theta2_without_bias = Theta2;
Theta2_without_bias(:,1) = [] ;% Deleting first column . Size = 10 x 25

sum_l1 = 0;
for i = 1 : size(Theta1_without_bias)(2),
	
	temp = Theta1_without_bias(:,i).^2;
	sum_int = 0;
	for j = 1 : size(temp)(1),
		sum_int = sum_int + temp(j);
	end
	sum_l1 = sum_l1 + sum_int;

end

sum_l2 = 0;
for i = 1 : size(Theta2_without_bias)(2),
	
	temp = Theta2_without_bias(:,i).^2;
	sum_int = 0;
	for j = 1 : size(temp)(1),
		sum_int = sum_int + temp(j);
	end
	sum_l2 = sum_l2 + sum_int;

end

reg_term_for_cost = sum_l1 + sum_l2;
reg_term_for_cost = lambda * reg_term_for_cost / ( 2 * m);

J = J + reg_term_for_cost;



%Making first column zero as they belong to bias term
Theta1(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m .* Theta1;


Theta2(:,1)= 0 ;
Theta2_grad = Theta2_grad + lambda / m .* Theta2;

end % if ends here

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
