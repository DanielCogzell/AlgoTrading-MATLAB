function out = roundTo(num , roundTo)
% function out = roundTo(num , roundTo)

% Copyright 2017 The MathWorks, Inc.

out = round(num ./ roundTo) .* roundTo;
