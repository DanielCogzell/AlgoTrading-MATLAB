function [returns , trades , classificationTree] = ...
    fxMachineLearningModel(xInSample , yInSample , xOutSample , yOutSample , tc , minPip)
% function [returns , trades , classificationTree] = ...
%     fxMachineLearningModel(xInSample , yInSample , xOutSample , yOutSample , tc , minPip)

% Copyright 2017 The MathWorks, Inc.

% Store the date
timeVector = xOutSample.Time;

% Take the factors and create out predictor table
xInSample = timetable2table(xInSample);
yInSample  = timetable2table(yInSample);
xInSample = xInSample(:,2:end);

% Now we can classify our responses...
futRet = yInSample.FutureReturn;

% Filter by min return greater than a pip
response = zeros(size(futRet));
response(futRet >  minPip * 0.0001) =  1;
response(futRet < -minPip * 0.0001) = -1;
response = categorical(response);

% Train the model
classificationTree = fitctree(...
    xInSample, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 4, ...
    'Surrogate', 'off', ...
    'ClassNames', [-1; 0; 1]);

% Run the model in the out-sample
xOutSample = timetable2table(xOutSample);
xOutSample = xOutSample(:,2:end);
yOutSample = timetable2table(yOutSample );
yOutSample = yOutSample.FutureReturn;  

% Run this trained model on the out of sample data
retPred = classificationTree.predict(xOutSample);

% Calculate the performance in the out of sample
positions=zeros(size(xOutSample,1), 1);
positions(retPred > 0) = 1;
positions(retPred < 0) = -1;

% Add trading cost only where we cross the spread
spreadCross = [false ; diff(positions) > 0];

returns = positions .* yOutSample - spreadCross .* tc;

trades = timetable(timeVector , positions , 'variablenames' , {'Trades'});
