function [X , y] = fxAlgoMakeSignals(price , nMinReturns , otherSignals , nMinFutReturns , downSample)
%function [X , y] = fxAlgoMakeSignals(price , nMinReturns , otherSignals , nMinFutReturns)
% Make the regression/input matrix X and the output vector y for our FX
% algo strategy.
%
% Inputs:
%       price - timetable of prices. I MAKE THE ASSUMPTION THAT THIS VARIABLE
%       IS OF ONE MINUTE BAR
%       nMinReturns - vector of nMinute returns that we want to use as the
%       predictor
%       otherSignals - function handles referring to other sigals
%       nMinFutReturns - nMinute future return

% Copyright 2017 The MathWorks, Inc.

allFactors = {};
factorNames = {};

for i = 1:length(nMinReturns)
    logCommand(sprintf('Creating %d Minute Returns' , nMinReturns(i)));
    allFactors{i} = nMinuteReturn(price , nMinReturns(i)); %#ok<*AGROW>
    factorNames{i} = ['Return_' num2str(nMinReturns(i))];
end
factorCount = length(allFactors) + 1;

% Add in the other signals?
for i = 1:length(otherSignals)
    
    fHand = str2func(otherSignals(i).fnHand);
    data  = price.(otherSignals(i).field);
    
    if ~isempty(otherSignals(i).params)
        logCommand(sprintf('Creating factor %s %d' , otherSignals(i).fnHand , otherSignals(i).params));
        allFactors{factorCount} = fHand(data , otherSignals(i).params);
        factorNames{factorCount} = [otherSignals(i).fnHand '_' num2str(otherSignals(i).params)];
    else
        logCommand(sprintf('Creating factor %s' , otherSignals(i).fnHand));
        allFactors{factorCount} = fHand(data);
        factorNames{factorCount} = otherSignals(i).fnHand;
    end
    factorCount = factorCount + 1;
    
end

% Create a new timetable
signal = timetable(price.Time , allFactors{:} , 'VariableNames' , factorNames);

% Merge with the other results
price = synchronize(price , signal);

% Create the future return
logCommand(sprintf('Calculating the %d minute future return' , nMinFutReturns));
futReturns = nMinuteReturn(price , nMinFutReturns);

% Stagger
futureReturn = NaN(size(futReturns,1) , 1);
futureReturn(1:end-nMinFutReturns) = futReturns(nMinFutReturns+1:end);

% Create a timetable for y
y = timetable(price.Time , futureReturn , 'VariableNames' , {'FutureReturn'});
price = synchronize(price , y);

% Downsample
logCommand(sprintf('Downsampling the data to %d minute bar' , nMinFutReturns));
if downSample
    price = price(1:nMinFutReturns:end,:);
end
price = rmmissing(price);

% Separate
X = price(:,1:end-1);
y = price(:,end);
logCommand('Finsished calculating factors and responses');

