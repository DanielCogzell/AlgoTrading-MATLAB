%% Test for the starting point sensitivity

%it will run the model for every minute possible in the first hour of the
%dataset

% Copyright 2017 The MathWorks, Inc.

currPair = 'EURUSD';
s = load(['.' filesep 'Data' filesep currPair]);
prices = s.prices;

%% Calculate our predictor/response
tradingFrequency = 60;

% These are the n-minute returns that we want to use to build our model
nMinReturns = [5 10 15 20 25 30 60];

% Other functions to use... Let's take a couple of toolbox functions

% MACD technical indicator
otherSignals = [];
otherSignals(1).fnHand = 'macd';
otherSignals(1).params = [];
otherSignals(1).field  = 'Mid';

% RSINDEX
for i = [5 10 15 20 25 30 60]
    otherSignals(end+1).fnHand = 'rsindex'; %#ok<SAGROW>
    otherSignals(end).params   = i;
    otherSignals(end).field    = 'Mid';
end

downSample = false;
[XX , yy] = fxAlgoMakeSignals(prices , nMinReturns , otherSignals , tradingFrequency , downSample);

%% Backtest
figure
drawnow
for startingPoint = 1:59
    
    % Set up our initial window of the first month. Note that we're using the
    % dateshift command to find the last day of the month
    inSampleStart = datetime('01-Jan-2007');
    outSampleStart = datetime('01-Feb-2008');
    outSampleEnd = datetime('28-Feb-2008');
    
    X = XX(startingPoint:tradingFrequency:end,:);
    y = yy(startingPoint:tradingFrequency:end,:);
    
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
    while outSampleStart < X.Time(end) && outSampleEnd < X.Time(end)
        
        % Get the in-Sample end date
        inSampleEnd = inSampleStart + calendarDuration(1,0,0) - caldays(1);
        
        % Get the in-sample
        tr = timerange(inSampleStart , inSampleEnd);
        xInSample = X(tr , 4:end);
        yInSample = y(tr , :);
        
        % Shift the timerange so we can get the out of sample data
        outSampleStart = inSampleEnd + caldays(1);
        outSampleStart = dateshift(outSampleStart , 'start' , 'month');
        outSampleEnd   = dateshift(outSampleStart , 'end' , 'month');
        
        % Get the out-sample
        tr = timerange(outSampleStart , outSampleEnd);
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
        inSampleStart = inSampleStart + calmonths(1);
        inSampleStart = dateshift(inSampleStart , 'start' , 'month');
        
        % Show progress
        logCommand(sprintf('Backtested from %s to %s' , char(outSampleStart) , char(outSampleEnd)));
        
    end
    
    hold on;
%     plot(allReturns.time , cumprod(1 + allReturns.RegReturns) , 'b');
    plot(allReturns.time , cumprod(1 + allReturns.MachineReturns));
    drawnow;
end
