%% Intelligence and cooperative strategy - ANN play IPD

clear all
close all
dbstop if error
profile on

tic

RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

%% set properties of networks

ninput = 2;
ninter = 10;
nmem = 10;
noutput = 1;

%% Set payoffs

PD=[1,8;2,5]; % payoff matrix
mem_cost = 0.01;

%% set GA properties

pop = 50; % population size
generations = 5000; % number of generations to run the GA for

mutrate = 0.1; % mutation rate per locus
mutsd = 0.5; % standard deviation of weight mutation
tmutsd = mutsd; % standard deviation of threshold mutation
wmax = 10;  % max/min range of each weight (i.e. gene)

% generate the first population of individuals and their weights
inputweights = -wmax + 2.*wmax.*rand(ninput,ninter,pop);
memweights = -wmax + 2.*wmax.*rand(ninter,pop); %zeros(nmem,ninter,pop);
interweights = -wmax + 2.*wmax.*rand(ninter,noutput,pop);
firstmove = round(rand(pop,1));
thresholds1 = -wmax + 2.*wmax.*rand(pop,10);
thresholds2 = -wmax + 2.*wmax.*rand(pop,1);
structure = floor(4.*rand(pop,2));
structure((structure(:,1)-structure(:,2))<0,2) = structure((structure(:,1)-structure(:,2))<0,1);
memyn = zeros(pop,10);
for i = 1:pop
    memyn(i,1:structure(i,2)) = 1;
end
structure(:,2) = [];

%% Storage matrices

resW = zeros(pop,generations); % matrix into which to write fitnesses
resfirstmove = zeros(pop,generations);
resintelligence = zeros(pop,generations);
resmem = zeros(pop,generations);
resstruc = zeros(pop,generations);

newinputw = zeros(size(inputweights)); % new set of input weights
newmemw = zeros(size(memweights)); % new set of memory weights
newinterw = zeros(size(interweights)); % new set of memory weights
newfirst = zeros(size(firstmove));   % new set of first moves
newthresh1 = zeros(size(thresholds1)); % new set of thresholds
newthresh2 = zeros(size(thresholds2)); % new set of thresholds
newstructure = zeros(size(structure));
W = zeros(1,pop); % fitnesses


%% generate test sets and test strategies

% set test set
rep = 10;
pp = 0:0.25:1;
p = zeros(1,length(pp).*rep.*20);
for i = 1:length(pp)
    p(1,1+(i-1)*rep*20:i*rep*20) = pp(i);
end
testset = rand(size(p)) <= p(:,:);

% test fixed strategy phenotypes
strats = [1,2,3,7,8];
nstrat = length(strats);
strat_phenotype = zeros(nstrat,length(p));

for i = 1:nstrat
    for j = 1:rep*length(pp)
        c = 0;
        for k = 1:20
            strat_phenotype(i,((j-1).*20)+1+c) = play(testset(1,((j-1)*20)+1:j*20),strat_phenotype(i,((j-1)*20)+1:j*20),strats(i),c);
            c = c+1;
        end
    end
end

stratmoves = zeros(pop,length(testset),nstrat);

for i = 1:nstrat
    stratmoves(:,:,i) = repmat(strat_phenotype(i,:),pop,1);
end

phenotype = zeros(pop,length(testset));

nearest_neighbour = zeros(1,generations);

dist2centre = zeros(1,generations);

distcentre = zeros(pop,generations);

coop = zeros(pop,generations);

nearest_strat = zeros(pop,generations);




%% run GA
Rounds = nbinrnd(1,0.02,(sum(1:pop).*generations),1) + 1; %zeros((sum(1:pop).*generations),1);
%Rounds(:,:) = 50;
C = 1;

for j = 1:generations
    
    disp(j)
    
    % get population stats
    
    % nnd
    
    for i = 1:pop
        for g = 1:rep*length(pp)
            c = 0;
            oppscore = 0;
            myscore = 0;
            memout = 0;
            for k = 1:20
                [X,memout] = playnet(oppscore,myscore,inputweights(:,:,i),memweights(:,i),interweights(:,:,i),thresholds1(i,:),thresholds2(i),memout,firstmove(i,1),c,structure(i,:),memyn(i,:));
                oppscore = PD(testset(1,((g-1)*20)+k)+1,X+1);
                myscore = PD(X+1,testset(1,((g-1)*20)+k)+1);
                c = c+1;
                phenotype(i,((g-1)*20)+k) = X;
            end
        end
    end
    
    ddd = distmatrix(phenotype);
    ddd = sort(ddd,1,'ascend');
    nearest_neighbour(j) = mean(ddd(2,:));
    
    % distance to centroid
    
    centroid = repmat(mean(phenotype,1),pop,1);
    dist2centre(1,j) = mean(sqrt(sum((phenotype-centroid).^2,2)),1);
    distcentre(:,j) = sqrt(sum((phenotype-centroid).^2,2));
    
    % nearest strategy
    
    stratdists = zeros(pop,nstrat);
    
    xxx = zeros(pop,nstrat);
    for i = 1:nstrat
        xxx(:,i) = sum((phenotype-stratmoves(:,:,i)).^2,2);
    end
    
    [Z,II] = min(xxx,[],2);
    nearest_strat(:,j) = II;
    
    % play nets against eachother
    
    w = zeros(pop,1);
    
    for i = 1:pop-1
        for k = i+1:pop
            memout1 = 0;
            memout2 = 0;
            score1 = 0;
            score2 = 0;
            rounds = Rounds(C);
            for g = 1:rounds
                c = g-1;
                if structure(i,1) == 0
                    X1 = firstmove(i,1);
                else
                    [X1,memout1] = playnet(score2,score1,inputweights(:,:,i),memweights(:,i),interweights(:,:,i),thresholds1(i,:),thresholds2(i),memout1,firstmove(i,1),c,structure(i,:),memyn(i,:));
                end
                if structure(k,1) == 0
                    X2 = firstmove(k,1);
                else
                    [X2,memout2] = playnet(score1,score2,inputweights(:,:,k),memweights(:,k),interweights(:,:,k),thresholds1(k,:),thresholds2(k),memout2,firstmove(k,1),c,structure(k,:),memyn(k,:));
                end
                score1 = PD(X1+1,X2+1);
                score2 = PD(X2+1,X1+1);
                w(i,1) = w(i,1) + score1./rounds;
                w(k,1) = w(k,1) + score2./rounds;
                coop(i,j) = coop(i,j) + X1./rounds;
                coop(k,j) = coop(k,j) + X2./rounds;
            end
            C = C+1;
        end
    end
    
    w = w./(sum(pop-1));
    %coop(j) = coop(j)./(rounds.*sum(1:pop-1));

    intelligence = sum(structure,2) + sum(memyn,2);
    
    w = w - (intelligence.*mem_cost);%((1-exp(-10.*(intelligence))).*mem_cost);
    w(w<0) = 0;
    
    resintelligence(:,j) = intelligence;
    resmem(:,j) = sum(memyn,2);
    resstruc(:,j) = sum(structure,2);
    resW(:,j) = w;
    
    % select reproducing individuals
    
    for i = 1:pop
        I = roulette(w);
        newinputw(:,:,i) = inputweights(:,:,I); % new set of input weights
        newmemw(:,i) = memweights(:,I); % new set of memory weights
        newinterw(:,:,i) = interweights(:,:,I); % new set of memory weights
        newfirst(i,1) = firstmove(I,1);   % new set of first moves
        newthresh1(i,:) = thresholds1(I,:);
        newthresh2(i,:) = thresholds2(I,:);
        newstructure(i,:,:) = structure(I,:);
    end
    
    inputweights = newinputw + (rand(size(newinputw))<mutrate).*randn(size(newinputw)).*mutsd;
    memweights = newmemw + (rand(size(newmemw))<mutrate).*randn(size(newmemw)).*mutsd;
    interweights = newinterw + (rand(size(newinterw))<mutrate).*randn(size(newinterw)).*mutsd;
    thresholds1 = newthresh1 + (rand(size(newthresh1))<mutrate).*randn(size(newthresh1)).*mutsd;
    thresholds2 = newthresh2 + (rand(size(newthresh2))<mutrate).*randn(size(newthresh2)).*mutsd;
    changefirst = rand(size(newfirst))<mutrate;
    newfirst(changefirst) = newfirst(changefirst).*(-1) + 1;
    firstmove = newfirst;
    changestruc = (rand(pop,2)<(mutrate./5)).*(2.*round(rand(pop,2))-1);
    changestruc((structure(:,1)+changestruc(:,1))-(sum(memyn,2)+changestruc(:,2))<0,2) = 0;
    changestruc((structure(:,1)+changestruc(:,1))<0,1) = 0;
    changestruc((sum(memyn,2)+changestruc(:,2))<0,2) = 0;
    changestruc((structure(:,1)+changestruc(:,1))>10,1) = 0;
    changestruc((sum(memyn,2)+changestruc(:,2))>10,2) = 0;
    %structure = structure + changestruc;
    
    for i = 1:pop
        if changestruc(i,1) == 1
            inputweights(:,structure(i,1)+1,i) = -wmax + 2.*wmax.*rand(ninput,1);
            interweights(structure(i,1)+1,:,i) = -wmax + 2.*wmax.*rand(1,noutput);
            structure(i,1) = structure(i,1) + 1;
        elseif changestruc(i,1) == -1
            J = floor((structure(i,1)).*rand)+1;
            if memyn(i,J) == 1
                if changestruc(i,2)==-1
                    changestruc(i,2) = 0;
                end
            end
            if J ~= structure(i,1)
                inputweights(:,J,i) = inputweights(:,structure(i,1),i);
                interweights(J,:,i) = interweights(structure(i,1),:,i);
                
                if memyn(i,structure(i,1)) == 1
                    memweights(J,i) = memweights(structure(i,1),i);
                    memyn(i,J) = 1;
                    memyn(i,structure(i,1)) = 0;
                    %structure(i,2) = structure(i,2) - 1;
                end
                
            else
                if memyn(i,J) == 1
                    memyn(i,J) = 0;
                    %structure(i,2) = structure(i,2) - 1;
                end
            end
            structure(i,1) = structure(i,1) - 1;
        end
        
        if changestruc(i,2) == 1
          J = floor((structure(i,1)-sum(memyn(i,:))).*rand) + 1;
          F = find(memyn(i,1:structure(i,1))==0);
          memweights(F(J),i) = -wmax + 2.*wmax.*rand;
          memyn(i,F(J)) = 1;
          %structure(i,2) = structure(i,2) + 1;
        elseif changestruc(i,2) == -1
            J = floor((sum(memyn(i,:))).*rand)+1;
            F = find(memyn(i,1:structure(i,1))==1);
            memyn(i,F(J)) = 0;
            %structure(i,2) = structure(i,2) - 1;
        end
    end
    
    coop(:,j) = coop(:,j)./(pop-1);
    
end


%% correlations
correlations = zeros(generations,1);

%resintelligence(:,i) = resintelligence(:,i)+0.001;

for i = 3:generations
    %[correlations(i),pvalues(i)] = corr(resintelligence(:,i),resW(:,i),'type','Kendall','rows','pairwise');
    correlations(i) = mean(resintelligence(:,i).*resW(:,i)) - (mean(resintelligence(:,i)).*mean(resW(:,i)));
end

yyy = zeros(5,generations);
for i=1:generations
    yyy(:,i) = histc(nearest_strat(:,i),1:5);
end

yyy = yyy./pop;

xxx = zeros(4,generations);
xxx(1,:) = yyy(1,:);
xxx(2,:) = yyy(2,:);
xxx(3,:) = yyy(3,:) + yyy(4,:);
xxx(4,:) = yyy(5,:);

dist2centre=dist2centre';
coopmean=sum(coop,1)./pop;
resint=(mean(resintelligence,1))';

%% Selection for cooperation

select_coop = zeros(generations,1);

[xresint,I] = sort(resint);

for i = 1:generations
    %[correlations(i),pvalues(i)] = corr(resintelligence(:,i),resW(:,i),'type','Kendall','rows','pairwise');
    select_coop(i) = mean(coop(:,i).*(resW(:,i)./mean(resW(:,i)))) - (mean(coop(:,i)).*mean(resW(:,i)./mean(resW(:,i))));
end

select_coop = select_coop(I);

r=ksrmv(xresint,select_coop,0.5);

plot(xresint,r.f,'-k')

%% plot results

% figure(1)
% subplot(2,1,1)
% plot(1:5000,coop,'.k')
% hold all
% set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',14,'TickLength',[.02 1])
% box off
% ylabel('Frequency of cooperation')
% subplot(2,1,2)
% harea=area(xxx');
% set(get(harea(1),'Children'),'FaceColor',[0 .3 0])
% set(get(harea(2),'Children'),'FaceColor',[0 0 0.4])
% set(get(harea(3),'Children'),'FaceColor',[0.4 0 0])
% set(get(harea(4),'Children'),'FaceColor',[1 0.8 0])
% set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',14,'TickLength',[.02 1])
% ylim([0 1])
% box off
% ylabel('Strategy frequency')
% xlabel('Generation')

figure(3)
subplot(2,1,1)
plot(1:generations,coopmean,'-k','linewidth',1)
hold all
set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',16,'TickLength',[.02 1])
box off
ylabel('Frequency of cooperation')
subplot(2,1,2)
harea=area(xxx');
ccc=[0 0 0; 1 1 1; 0.4 0.4 0.4; 0.8 0.8 0.8];
colormap(ccc)
set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',16,'TickLength',[.02 1])
ylim([0 1])
l=legend(harea,'AllD','AllC','Tit-fortat','Pavlov','Orientation','horizontal');
box off
ylabel('Strategy frequency')
xlabel('Generation')


figure(2)

subplot(3,1,1)
plot(coopmean,correlations,'o','Color',[0.2 0.2 0.2])
hold all
%plot(0:0.01:0.8,(4.18946818661838).*(0:0.01:0.8).^3 + (-8.16736405121770).*(0:0.01:0.8).^2 + (4.34850411432926).*(0:0.01:0.8) + (-0.369396381408357),'-k','linewidth',1)
set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',14,'TickLength',[.02 1])
box off
ylabel('Correlation coefficient')
xlabel('Frequency of cooperation')
%xlim([0 0.8])
%ylim([-0.5 1])
title('(a)')

subplot(3,1,2)
plot(dist2centre,correlations,'o','Color',[0.2 0.2 0.2])
hold all
%plot(0:0.01:16,(-0.00197790193779445).*(0:0.01:16).^2 + (0.071500688344131).*(0:0.01:16) + (-0.370921018641865),'-k')
%plot(0:0.01:16,([0.0376664341060592;]).*(0:0.01:16) + (-0.253654434826855),'-k','linewidth',1)
set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',14,'TickLength',[.02 1])
box off
ylabel('Correlation coefficient')
xlabel('Distance to centroid')
xlim([0 16])
%ylim([-0.5 1])
title('(b)')

subplot(3,1,3)
plot(resint,correlations,'o','Color',[0.2 0.2 0.2])
hold all
%plot(0:0.001:0.35,([225.843153914002;]).*(0:0.001:0.35).^3 + ([-112.355047899833;]).*(0:0.001:0.35).^2 + ([13.8624103476047;]).*(0:0.001:0.35) + ([-0.240614362256628;]),'-k','linewidth',1)
set(gca,'linewidth',1,'XMinorTick','on','YMinorTick','on','FontName','Arial','FontSize',14,'TickLength',[.02 1])
box off
ylabel('Correlation coefficient')
xlabel('Intelligence')
%xlim([0 0.35])
%ylim([-0.5 1])
title('(c)')

toc

profile viewer