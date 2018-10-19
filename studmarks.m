%% Acedemic Performance Prediction Model Using Machine Learning
%  
%We've used two different methods to predict the student's SGPA in the 
%the consequent semesters. 
%Feature Normalization And Gradient Descent
%
%
% 
fprintf('\tIn this part, we will implement Linear Regression to\n');
fprintf('\tpredict the SGPA of students in consequent semesters.\n');   
fprintf('\tThe file')
fprintf('ex11.txt\n')
fprintf('\tcontains a training set of student SGPA in previous Semesters\n')
fprintf('\tThe first column is the SGPA, the\n')
fprintf('\tsecond column is the SGPA, and the third column\n');
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex11.txt');
X = data(:, [1,2]);
y = data(:, 3);
m = length(y);
plotData(X,y);
% Print out some data points
fprintf('First 10 examples from the dataset: \n\n\n');
fprintf('First Sem SGPA \t\t\tSecond Sem SGPA\t\t\t Third Semgpa\n\n');

fprintf(' %f \t\t\t%f\t\t%f\n', [X([1:10],:)  y(1:10,:)]'');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');




% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Our task is to first make sure that the functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that,we will try running gradient descent with 
%               different values of alpha and see which one gives
%               us the best result.
%
%               Finally, we complete the code at the end
%               to predict the SGPA in the consequent semester. 
%


fprintf('Using Gradient Descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(featureNormalize(X), featureNormalize(y), theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('\nEnter the marks in First Sem ');
FirstSemMarks= input('');
fprintf('\nEnter the marks in SEC Sem ');

SecSemMarks=input('');
v=[0 (FirstSemMarks-mean(X(:,2)))/std(X(:,2)) (SecSemMarks-mean(X(:,3)))/std(X(:,3))];
% Estimate the marks.
thrdSemMarks=v*theta;
thrdSemMarks=thrdSemMarks*std(y)+mean(y);


% ============================================================

fprintf(['Predicted marks in 3rd Sem are :\n $%f\n\n\n'], thrdSemMarks);

fprintf('Program paused. Press enter to continue.\n');
pause;
hold on;
plot(X(:,[1,2]), X*theta, '-')
theta=normalEqn(X,y);
thrdSemMarks=[1 FirstSemMarks SecSemMarks]*theta;
fprintf('USING NORMAL EQUATIONS\n\n');
fprintf(['Predicted marks in 2rd Sem are :\n $%f\n'], thrdSemMarks);



% ============================================================



