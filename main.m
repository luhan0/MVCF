clear;close all;
addpath('.\funs');
addpath('.\datasets');
% ===================== Parameters Setting =============================
dataname='MSRC';
anchor_rate = 0.7;
All_r = 1.5;

% % >>> Ring Experiments >>> 
% dataname='threeR';
% anchor_rate = 0.5;
% All_r = 12;

% =============== initialize =============================
fprintf('----------【%s】-------------\n',dataname);
load([dataname '.mat']);
try
    gt=double(Y);%r=1.8
catch
end
cluser_num=length(unique(gt));
nV=size(X,2);
[nN,~] = size(X{1});%[样本维数，总样本数目]
each_cluster_num = nN/cluser_num;%每一类的数目
for r=All_r
    tic
    alpha=1/nV*ones(1,nV);
    opt1. style = 1;
    opt1. IterMax = 150;
    opt1. toy = 0;
    H = cell(1, nV); 
    S = cell(1, nV);
    Tri = cell(1, nV);
    [~, C] = FastmultiCLR(X,cluser_num,anchor_rate, opt1,10);
    nM = floor(nN*anchor_rate);
    for v = 1: nV
        for i=1:nM
            Tri{v}(i,i)=sum(C{v}(:,i));
        end
        S{v}=C{v}/Tri{v}*(C{v})'+eps;
        H{v}=-log(S{v});
        H{v} = H{v}-diag(diag(H{v}));
    end
    % Y_consturction
    Y=zeros(nN,cluser_num);
    for i=1:nN
        Y(i,mod(i,cluser_num)+1)=1;
    end
    obj=[1];flag=1;iter=1;
% ===================== iter =======================
    while flag==1   
        % Solving Y
        D_sum=zeros(nN,nN);
        for v = 1:nV
            D_sum = D_sum +(alpha(v)^r)*H{v};
        end
            for i = 1:nN 
                M = ((D_sum(i,:)*Y))';%*lambda
                [~, m] = min(M);
                Y(i,:) = 0;
                Y(i,m) = 1;
            end
        % Solving alpha(v)
            o=0;
            for i=1:v
                h(i)=trace(Y'*H{i}*Y);
                temp(i)=((h(i))^(1/(1-r)));
            end
            for i=1:v
                alpha(i)=temp(i)/sum(temp);
            end
            
            obj=[obj trace(Y'*D_sum*Y)];
            iter=iter+1;
            if abs(obj(iter)-obj(iter-1))<10e-8 ||iter>50%
                flag=0;
            end
    end
    time = toc;
    [~, label] = max(Y');
    result = ClusteringMeasure(gt,label);
    fid=fopen(['result__' dataname '.txt'],'a');
    fprintf(fid ,'MultiK-sum %4f %4f %4f iter=%d r=%3f time=%4f\n',result(1:3),iter,r,time);
    fprintf('MultiK-sum:%4f,%4f,%4f iter=%d r=%3f time=%4f\n',result(1:3),iter,r,time);
    fclose(fid);
 end