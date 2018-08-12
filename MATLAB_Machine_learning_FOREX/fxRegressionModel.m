function [returns , trades] = fxRegressionModel(xInSample , yInSample , xOutSample , yOutSample , tc)
% function [returns , trades] = fxRegressionModel(xInSample , yInSample , xOutSample , yOutSample , tc)

% Copyright 2017 The MathWorks, Inc.

xInSample  = xInSample{:,:};
yInSample  = yInSample{:,:};
xOutSample = xOutSample{:,:};
yOutSample = yOutSample{:,:};

modelTrain = fitlm(xInSample , yInSample , 'linear');

% Run this trained model on the out of sample data
retPred = predict(modelTrain , xOutSample);

% Calculate the performance in the out of sample
positions=zeros(size(xOutSample,1), 1);
positions(retPred > tc)=1;
positions(retPred < -tc)=-1;

% Add trading cost only where we cross the spread
spreadCross = [false ; diff(positions) > 0];

returns = positions .* yOutSample - spreadCross .* tc;

trades = positions;
