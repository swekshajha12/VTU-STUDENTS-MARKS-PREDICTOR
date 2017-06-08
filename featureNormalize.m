function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

[m,n]=size(X);
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
mu=mean(X);

for i=1:n
  X(:,i)=X(:,i)-mu(i);
end
sigma = std(X);

for i=1:n
  if(sigma(i)!=0)
    X(:,i)=X(:,i)/sigma(i);
   end 
end
X_norm=X;
mu=mean(X);
sigma = std(X);
 