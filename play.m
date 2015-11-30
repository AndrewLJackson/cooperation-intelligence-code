function x=play(opp,me,strategy,c) 
SotonM=[1,0,1,1,0,1,1,1];
SotonS=[0,1,0,0,1,0,0,0];
if(strategy==1) %always defect
    x=0;
elseif(strategy==2) %always cooperate
    x=1;
elseif(strategy==3) %titfortat
    if(c == 0)
        x=1;
    else
        x=opp(c);
    end
elseif(strategy==4) %GRIM
    if(c == 0)
        x=1;
    else
        if(sum(opp) < c)
            x=0;
        else
            x=1;
        end
    end
elseif(strategy==5) %Soton Master
    if(c == 0)
        x=SotonM(1);
    elseif(c<8&&c>0)
        if(opp(1:c)==SotonS(1:c)) %Opponent has so far signalled correctly as a slave
            x=SotonM(c+1); %Give the next signal
        elseif(opp(1:c)==SotonM(1:c)) %Opponent has so far signalled as a master
            x=SotonM(c+1); %Give the next signal
        else
            x=0; %Otherwise, cease signalling and sabotage opponent
        end
    else
        if(opp(1:8)==SotonS) %found a slave
            x=0; %exploit
        elseif(opp(1:8)==SotonM) %found another master
            x=1; %cooperate
        else 
            x=0; %Not a Soton program; sabotage
        end
    end
elseif(strategy==6) %Soton slave
    if(c == 0)
        x=SotonS(1);
    elseif(c<8&&c>0)
        if(opp(1:c)==SotonS(1:c)) %Opponent has so far signalled correctly as a slave
            x=SotonS(c+1); %Give the next signal
        elseif(opp(1:c)==SotonM(1:c)) %Opponent has so far signalled as a master
            x=SotonS(c+1); %Give the next signal
        else
            x=0; %Otherwise, cease signalling and sabotage opponent
        end
    else
        if(opp(1:8)==SotonS) %found a slave
            x=1; %cooperate
        elseif(opp(1:8)==SotonM) %found a master
            x=1; %self-sacrifice
        else 
            x=0; %Not a Soton program; sabotage
        end
    end
    
elseif(strategy==7) % tit fot two tats
    if(c == 0)
        x=1;
    elseif(c==1)
        x=1;
    elseif(c>1)
        if (opp(c)==0) && (opp(c-1)==0)
            x=0;
        else
            x=1;
        end
    end
    
    
elseif(strategy==8) % Pavlov
    if(c == 0)
        x=1;
    elseif opp(c) == 1
        x=me(c);
    else
        x=abs((me(c))-1);
    end
else %50/50
    s=rand;
    if(s<0)
        x=0;
    else
        x=1;
    end
end