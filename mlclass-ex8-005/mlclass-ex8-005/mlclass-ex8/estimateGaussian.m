function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%Finding mean
temp = sum(X); %sum of all fectures separately in a row vector
temp = temp';
mu = temp / m;

tempX = X;
%Finding variance
for i = 1:n,
	tempX(:,i) = X(:,i).- mu(i);
end

tempX = tempX.^2;
tempX = sum(tempX);
tempX = tempX';
sigma2 = tempX / m;











% =============================================================


end
