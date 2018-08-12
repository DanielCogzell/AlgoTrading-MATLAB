%% Machine Learning for FX
%
% Moving on from regression, let's take a look at trying to improve the
% results of our trading using machine learning techniques.

% Copyright 2017 The MathWorks, Inc.

%% FX Factors
%
% Load in the factor matrix from the previous step, Demo1_fxRegressionDemo
load('fxData');

%% Data subset
%
% Prepare the data as before. Take our factor table and take a subset for
% training purposes using a timerange object
tr = timerange('2007-01-01' , '2007-02-28');
predictorInSample = timetable2table(X(tr , :) , 'ConvertRowTimes' , false);
responseInSample  = timetable2table(y(tr , :) , 'ConvertRowTimes' , false);
tIn = X.Time(tr);

% Take the factors and create out predictor table
predictorTable = predictorInSample(:,4:end);
clear('predictorInSample');

%% Responses
% Let's categorise the responses as buy, sell or hold. We will use the
% value of the return to classify the responses in our supervised learning
% problem.
%
% We want to classify the problem into three different states. We will use
% the median bid/offer spread to define a minimum predicted return at which
% we wouldn't trade.
minRet = nanmedian(prices.Ask - prices.Bid);    %cost to trade

% Let's round this UP to the nearest pip...
minRet = roundUpTo(minRet , 0.0001);   

% Now we can classify our responses...
futRet = responseInSample.FutureReturn;

% Filter by min return
response = zeros(size(futRet));     %creating hold signals
response(futRet > minRet) = 1;      %buy signals when futRet > mid spread
response(futRet < -minRet) = -1;    %sell

% Add the response to the table
predictorTable.response = response; 

%% Machine Learning
%
% Let's open up our machine learning tool from the Statistics and Machine
% Learning toolbox. From the list of variables, make sure you
%
% 1. Choose the variable predictorTable
% 2. Set the response variable to be the response
% classificationLearner

%% Classification Tree
%
% Let's choose one of the methods. In this case, we aren't choosing the
% best performing method, but we choose one that's easy to interpret and
% explain, namely the classification tree
cTree = fitctree(...
    predictorTable(:,1:end-1), ...
    predictorTable(:,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 4, ...
    'Surrogate', 'off', ...
    'ClassNames', [-1; 0; 1]);

%% Visualise the tree
view(cTree , 'Mode' , 'graph')

%% Test the results in sample...
%
% We send the original predictor table into the classification tree to get
% the predicted returns, and take a position long/short when the predicted
% class is non-zero
%
% When we look at the returns in-sample, we see a similar performance graph
% when we compare to the two regression techniques
retPredictionML = cTree.predict(predictorTable(:,1:end-1));

positions = zeros(size(futRet));
positions(retPredictionML > 0) = 1;
positions(retPredictionML < 0) = -1;

actualReturns = positions .* futRet;
inSampleRegressionReturns = cumprod(1+actualReturns);

f1 = findobj('tag' , 'insamplefigure');
if exist('f1' , 'var') & isa(f1 , 'handle') & isvalid(f1)
    figure(f1)
    plot(tIn , inSampleRegressionReturns , 'g');
    legend('Linear Regression' , 'Stepwise' , 'Machine Learning')
else
    f1 = figure;
    plot(tIn , inSampleRegressionReturns , 'g');
end
title('In-Sample Results');

%% And for the out-sample...
%
% Repeat the process of generating the out-sample subset using a timerange
% object, and sending this into the classification tree to get a prediction
% for future returns, which we then apply against the true future returns.
%
% On charting it, we see an improvement in the results for the machine 
% learning technique, and we will continue with both the regression
% and the machine learning technique
tr = timerange('2007-03-01' , '2007-03-31');
predictorOutSample = timetable2table(X(tr , :) , 'ConvertRowTimes' , false);
responseOutSample  = timetable2table(y(tr , :) , 'ConvertRowTimes' , false);
tOut = X.Time(tr);
futRet = responseOutSample.FutureReturn;

% Take the factors and create out predictor table
predictorTable = predictorOutSample(:,4:end);

% Run our prediction of our classification tree
retPredictionML = cTree.predict(predictorTable(:,1:end));

% Gather the returns
positions = zeros(size(futRet));
positions(retPredictionML > 0) = 1;
positions(retPredictionML < 0) = -1;

actualReturns = positions .* futRet;
outSampleRegressionReturns = cumprod(1+actualReturns);

f2 = findobj('tag' , 'outsamplefigure');
if exist('f2' , 'var') & isa(f2 , 'handle') & isvalid(f2)
    figure(f2)
    plot(tOut , outSampleRegressionReturns , 'g');
    legend('Linear Regression' , 'Stepwise' , 'Machine Learning')
else
    f2 = figure;
    plot(tOut , outSampleRegressionReturns , 'g');
end
title('Out-Sample Results');

