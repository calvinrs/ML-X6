function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
Ss = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

answerM = zeros(size(Cs,1)*size(Ss,1),3);
k = 0;

for i = 1:size(Cs,1)
    
    for j = 1:size(Ss)
       
       k = k+1;
       thisC = Cs(i);
       thisSigma = Ss(j);

       answerM(k,1) = thisC;
       answerM(k,2) = thisSigma;
       
       model= svmTrain(X, y, thisC, @(x1, x2) gaussianKernel(x1, x2, thisSigma));

       predictions = svmPredict(model, Xval);
       
       answerM(k,3) = mean(double(predictions ~= yval));
       
       
       
    end
    
end

[minval, mindex] = min(answerM(:,3));

C = answerM(mindex,1);
sigma = answerM(mindex,2);    








% =========================================================================

end
