function returns = nMinuteReturn(tt , nMins)
%function out = nMinuteReturn(tt , nMins)
%
% Given the time series of one minute bar data, create the return...

% Copyright 2017 The MathWorks, Inc.

data = tt.Mid;
returns = NaN(size(data,1),1);
returns(nMins+1:end) = data(nMins+1:end) ./ data(1:end-nMins) - 1;

