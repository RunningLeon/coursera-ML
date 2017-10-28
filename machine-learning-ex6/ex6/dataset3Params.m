function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the resultsl (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the resultsl C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
params = [0.01 0.03 0.1 0.3 1 3 10 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the resultsl C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
len = length(params);
results = zeros(len*len,3);
k = 0;
for i = 1:len
  C = params(i);
  for j = 1:len
    sigma = params(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    k += 1;
    results(k,:) = [C sigma err];
  endfor
endfor

% sort matrix by columns # 3
results = sortrows(results, 3);
C = results(1, 1);
sigma = results(1, 2);

fprintf('C %.4f, sigma: %.4f',C,sigma);




% =========================================================================

end
