function out = roundUpTo(num , roundTo)
%function out = roundUpTo(num , roundTo)
%

% Copyright 2017 The MathWorks, Inc.

out = ceil(num ./ roundTo) .* roundTo;
