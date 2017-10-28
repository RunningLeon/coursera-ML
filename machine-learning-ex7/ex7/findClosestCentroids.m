function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);
% method one with two loops

%for i = 1:m
%  x_i = X(i,:);
%  distances = zeros(K,1);
%  for j = 1:K
%    centroid_j = centroids(j,:);
%    dis = 1/m*sum((x_i - centroid_j).^2);
%    distances(j) = dis;
%  endfor
%  [val k] = min(distances);
%  idx(i) = k; 
%endfor

% method two with one loop
J = zeros(m,K);
for i = 1:K
  % create a m x size(centroids,2) matrix with centroid(i,:) repeated m times;
  repCentroids = repmat(centroids(i,:), m, 1);
  % add Ji of ith centroids and X to J
  J(:, i) = 1/m*sum((X - repCentroids).^2,2);
endfor

% idxT is a vector contains the indices of min value in each column of J transpose;
[val idxT] = min(J');
idx = idxT';

% fprintf("size idx:%f",size(idx)); 






% =============================================================

end

