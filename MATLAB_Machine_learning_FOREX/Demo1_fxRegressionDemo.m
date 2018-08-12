%% FX regression/stepwise/machine learning backtest
%
% This demo shows how to apply some simple regression and machine learning
% techniques to intraday FX trading. In our demo, we want to attempt to
% predict the 60 minute future return of a particular FX pair using a
% number of techniques
%
% * Linear Regression
% * Stepwise Regression
% * Machine Learning (specifically a classification tree)
%
% We will test this across one of the most liquid FX pairs, namely
% eurodollar.

% Copyright 2017 The MathWorks, Inc.

%% Data
%
% The data we have is ten years worth of one-minute bar prices for a series
% of currency pairs.
% We will take EURUSD as the pair in question.
currPair = 'EURUSD';
s = load(['Data' filesep currPair]);
prices = s.prices;

%% Plot
%
% The data is stored using the timetable construct, and the variable prices
% has three columns, Bid Ask and Mid.
figure;
plot(prices.Time , prices.Mid)
title(['Currency Pair ' currPair]);
box on;

%% Calculate our predictor/response
%
% First, we want to create a function that generates our predictor/response
% tables which we can then use for regression or machine learning.
%
% We will take a number of different factors in our predictive model
%
% * N-Minute returns (over various window lengths)
% * Moving Average Convergence/Divergence
% * Relative Strength Index (over various window lengths)

% Define the traing frequency of 60 minutes
tradingFrequency = 60;

% These are the n-minute returns that we want to use to build our model
nMinReturns = [5 10 15 20 25 30 60];

% Other functions to use... Let's take a couple of toolbox functions

% MACD technical indicator
otherSignals = [];
otherSignals(1).fnHand = 'macd';    %function handle initial
otherSignals(1).params = [];        %parameter initial
otherSignals(1).field  = 'Mid'; 

% RSINDEX
for i = [5 10 15 20 25 30 60]
    otherSignals(end+1).fnHand = 'rsindex'; %This is the hard coded signals
    otherSignals(end).params   = i;         %time horizons to use
    otherSignals(end).field    = 'Mid';     
end

[X , y] = fxAlgoMakeSignals(prices , nMinReturns , otherSignals , tradingFrequency , true);

%% Get an in-Sample
%
% Select one month of data - to do this, let's use the new timerange
% object which allows us to quickly window into our timetable to select the
% time period we want.

% The in-sample range we want to take is two months worth of data
tr = timerange('2007-01-01' , '2007-02-28');

XInSample = timetable2table(X(tr , 4:end) , 'convertrowtimes' , false);
yInSample = timetable2table(y(tr , :) , 'convertrowtimes' , false);
tIn = X.Time(tr);

%% Train a linear model
%
% We use the statistics toolox fitlm to fit a linear model, and we can use
% the predict method with the 
modelTrain = fitlm([XInSample yInSample] , 'linear')    %training for linear regression model

%% Run the in-sample prediction
retPredictionRegress = predict(modelTrain , XInSample); %Standard linear regression prediction

%% View results in sample by back testing
%NOTE THAT THIS IS (IN SAMPLE) and so the backtesting should yield good
%returns if the model was decent. Out of sample is the true test.
%
% We calculate the returns by simply multiplying the actual returns by the
% sign of the predicted returns, then apply a cumprod to generate a return
% curve
positions = zeros(size(retPredictionRegress));
positions(retPredictionRegress > 0) = 1;        %whenever the prediction is > 0, go long
positions(retPredictionRegress < 0) = -1;       % // // // go short

actualReturns = positions .* yInSample{:,1};    %backtesting part.
inSampleRegressionReturns = cumprod(1+actualReturns);

f1 = figure('tag' , 'insamplefigure');
plot(tIn , inSampleRegressionReturns);
title('In-Sample Results');

%% Out sample results
% Real test (Out of sample) using all the variables given.
% Take our predictive model and apply it to the out sample - in this case
% we are taking an out of sample to be one month.
%
% The performance isn't great in the out-sample, but let's persevere.
tr = timerange('2007-03-01' , '2007-03-31');
XOutSample = timetable2table(X(tr , 4:end) , 'convertrowtimes' , false);
yOutSample = timetable2table(y(tr , :) , 'convertrowtimes' , false);
tOut = X.Time(tr);

retPred = predict(modelTrain , XOutSample);

positions=zeros(size(XOutSample,1), 1);
positions(retPred > 0)=1;
positions(retPred < 0)=-1;

actualReturns = positions .* yOutSample{:,1};
outSampleRegressionReturns = cumprod(1 + actualReturns);

f2 = figure('tag' , 'outsamplefigure');
plot(tOut , outSampleRegressionReturns)
title('Out-Sample Results');


%% Repeat this process using a stepwise 
%
% Using the stepwiselm function, we can re-run this and (allow the stepwise
% algorithm to discard terms.)
modelStepwise = stepwiselm([XInSample yInSample] , 'linear' , 'upper' , 'linear')

%% Predict In-Sample results
retPrediction = predict(modelStepwise , XInSample);

positions = zeros(size(retPrediction));
positions(retPrediction > 0) = 1;
positions(retPrediction < 0) = -1;

actualReturns = positions .* yInSample{:,1};
inSampleStepwiseReturns = cumprod(1+actualReturns);

figure(f1); hold on
plot(tIn , inSampleStepwiseReturns , 'r');
legend({'Linear Regression' , 'Stepwise'});


%% Run for our out of sample
retPred = predict(modelStepwise , XOutSample);
positions=zeros(size(retPred,1), 1);
positions(retPred > 0)=1;
positions(retPred < 0)=-1;

actualReturns = positions .* yOutSample{:,1};

outSampleStepwiseReturns = cumprod(1 + actualReturns);

figure(f2)
hold on
plot(tOut , outSampleStepwiseReturns , 'r')
legend({'Linear Regression' , 'Stepwise'});

%% Save the results for stage #2, machine learning...
warning('off');
save('fxData' , 'prices' , 'X' , 'y');
% clear;


