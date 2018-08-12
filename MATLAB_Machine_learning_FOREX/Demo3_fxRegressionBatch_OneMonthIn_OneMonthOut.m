%% Backtest
%
% We will now backtest this model over the entire time history of available
% data.
%
% For speed purposes, we will only compare the regression model against the
% classification tree. Once again, we will use the timerange object to
% window into our data, fitting both of the models for our in-sample (of
% one month) then trading this model for the out-sample (also one month)

% Copyright 2017 The MathWorks, Inc.

%% Load in the data
%
% Load in the pre-saved results from Demo1, which has all the derived
% factors
logCommand('Loading historical results')
load('fxData');

%% Batch process...
%
% Set up our initial window of the first month. Note that we're using the
% dateshift command to find the last day of the month
startDate = datetime('01-Jan-2007');
endDate   = dateshift(startDate , 'end' , 'month');

%% Run the backtest

% Create an empty timetable to store all the results
allReturns = timetable;

% Get the in-sample subset
tr = timerange(startDate , endDate);
xInSample = X(tr , 4:end);
yInSample = y(tr , :);

logCommand('Starting Backtest');

regTrades  = [];
mlTrades   = [];

% Are we implementing a trading cost into our backtest?
tradingCost = true;
if tradingCost
    % Estimate the trading cost as the median size of the bid-offer spread
    tc = nanmedian(prices.Ask - prices.Bid) / nanmean(prices.Mid);
else
    tc = 0;
end

% Minimum pip return for the classification
minPip = 1.5;

% Use a while loop to go throuh our data
while startDate < X.Time(end) && endDate < X.Time(end)
    
    % Shift the timerange so we can get the out of sample data
    startDate = startDate + calmonths(1);
    endDate   = dateshift(startDate , 'end' , 'month');
    
    % Get the out-sample
    tr = timerange(startDate , endDate);
    time = X.Time(tr);
    xOutSample = X(tr , 4:end);
    yOutSample = y(tr , :);
    
    % Train the models
    [thisMonthRegressionReturns , thisMonthRegTrades] ...
        = fxRegressionModel(xInSample , yInSample , xOutSample , yOutSample , tc);
    [thisMonthMachineReturns,thisMonthMLTrades,tree] ...
        = fxMachineLearningModel(xInSample , yInSample , xOutSample , yOutSample , tc , minPip);
    
    %Store the trades - we can do some analysis on these if we want
    regTrades = [regTrades ; thisMonthRegTrades]; %#ok<*AGROW>
    mlTrades = [mlTrades ; thisMonthMLTrades]; %#ok<*AGROW>
    
    % Store this month's returns
    thisMonthReturns = timetable(...
        time , ...
        thisMonthRegressionReturns , ...
        thisMonthMachineReturns , ...
        'variablenames' , {'RegReturns' , 'MachineReturns' });
    
    % Stack them up
    allReturns = [allReturns ; thisMonthReturns];
    
    % Shift the month on - the new in-sample is the old out-sample
    xInSample = xOutSample;
    yInSample = yOutSample;
    
    % Show progress
    logCommand(sprintf('Backtested from %s to %s' , char(startDate) , char(endDate)));
    
end

figure;
hold on;
plot(allReturns.time , cumprod(1 + allReturns.RegReturns) , 'b');
plot(allReturns.time , cumprod(1 + allReturns.MachineReturns) , 'g');
legend('Regression' , 'Classifiation Tree' , 'location' , 'nw');
title('Backtest Results: 1 Month In-Sample, 1 Month Out-Sample');


