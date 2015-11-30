function [X,memout] = playnet(oppscore,myscore,win,wmem,wout,threshold1,threshold2,memin,turn1,c,structure,memyn)

win = win(:,1:structure(1));
wmem = wmem';
memyn = memyn(:,1:structure(1));


if(c == 0)
    X = turn1;
    memout = 0;
    
elseif c == 1
    inter = oppscore.*win(1,:) + myscore.*win(2,:);
    inter = 1 ./ (1+exp(-(inter-threshold1(1:structure(1)))));
    memout = inter;
    Y = 1 ./ (1+exp(-(sum(inter.*(wout(1:structure(1))'))-threshold2)));
    X = rand<Y;
    
else
    if structure(2)==0
        inter = oppscore.*win(1,:) + myscore.*win(2,:);
        inter = 1 ./ (1+exp(-(inter-threshold1(1:structure(1)))));
        memout = 0;
        Y = 1 ./ (1+exp(-(sum(inter.*(wout(1:structure(1))'))-threshold2)));
        X = rand<Y;
    else
        inter = oppscore.*win(1,:) + myscore.*win(2,:) + (memin.*wmem(1:structure(1)).*memyn);
        inter = 1 ./ (1+exp(-(inter-threshold1(1:structure(1)))));
        memout = inter;
        Y = 1 ./ (1+exp(-(sum(inter.*(wout(1:structure(1))'))-threshold2)));
        X = rand<Y;
    end
end




end