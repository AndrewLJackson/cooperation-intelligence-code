function [I] = roulette(x)
% selects an entry from array x by roulette.

%cx = cumsum(x)./sum(x);
% [I] = min(find(rand<=cx));
[I] = find(rand<=cumsum(x)./sum(x), 1 );