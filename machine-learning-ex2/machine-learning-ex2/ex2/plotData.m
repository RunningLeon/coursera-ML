function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos_idx = find(y == 1);
neg_idx = find(y == 0);
plot(X(pos_idx,1),X(pos_idx,2),'k+', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'MarkerSize',8) 
hold on;
plot(X(neg_idx,1),X(neg_idx,2),'ko', 'LineWidth', 2, 'MarkerFaceColor', 'y', 'MarkerSize',8) 
xlabel('Exam 1 score'); % Set the y-axis label
ylabel('Exam 2 score'); % Set the x-axis label
title('Scatter plot of training data'); % Set title
legend('Admitted', 'Not admitted');  % Set legend







% =========================================================================



hold off;

end
