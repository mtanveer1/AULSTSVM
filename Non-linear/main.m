clc;
clear all;
close all;
file1 = fopen('result.txt','a+');

max_trial =10;
global p1 p2;
% ep = 0.9;
            
for load_file = 1:2
    %% initializing variables
no_part=5;
    %% to load file
    switch load_file
% Add datasets here
       case 1
            file = 'votes';
            test_start =301;
            cvs1=0.001;
            avs1=0.5;
            mus=4;

       case 2
            file = 'vowel';
            test_start =501;
            cvs1=100;
            avs1=0.1;
            mus=4;

        otherwise
            continue;
    end

uvs1=0.15;

         epsv=1;
%Data file call from folder   
filename = strcat('newd/',file,'.txt');
    A = load(filename);
    [m,n] = size(A);
%define the class level +1 or -1    
    for i=1:m
        if A(i,n)==0
            A(i,n)=-1;
        end
    end
% Dividing the data in training and testing    
test_start=m/2;
  [no_input,no_col] = size(A);
    test = A(test_start:m,:);
    train = A(1:test_start-1,:);
    x1 = train(:,1:no_col-1);
    y1 = train(:,no_col);
	    
    [no_test,no_col] = size(test);
    xtest0 = test(:,1:no_col-1);
    ytest0 = test(:,no_col);

    %% Universum
    A=[x1 y1];
    [no_input,no_col] = size(A);
   obs = A(:,no_col);   
%     C=A;
    C1= A(1:test_start-1,:);
    A = [];
 B = [];

for i = 1:test_start-1
    if(obs(i) == 1)
        A = [A;C1(i,1:no_col-1)];
    else
        B = [B;C1(i,1:no_col-1)];
    end;
end;
 u=ceil(uvs1*(test_start-1));
sb1=size(A,1);
sb=size(B,1);
ptb1=sb1/u;
ptb=sb/u;
Au=A(1:ptb1:sb1,:);
Bu=B(1:ptb:sb,:);
di=size(Au,1)-size(Bu,1);
if(di>0)
Bu=[Bu ;Bu(1:abs(di),:)];
elseif(di<0)
Au=[Au ;Au(1:abs(di),:)];
end   
 U=(Au+Bu)/2; 
   
    A1=A;
    %Combining all the column in one variable
     A=[x1 y1];    %training data
    A_test=[xtest0,ytest0];    %testing data
 %% initializing crossvalidation variables
    A=A_test;
    [lengthA,n] = size(A);
    min_err = -10^-10.;
 
i=1;
% p1=sqrt(sum1)/lengthA;
   
 for u1 = 1:length(uvs1)
            u_per = uvs1(u1)   

  for C1 = 1:length(cvs1)
            c = cvs1(C1)
%         for C2 = 1:length(cvs1)
%             c2 = cvs1(C2)
            c2=c;
        for An1 = 1:length(avs1)
            a = avs1(An1)
%             c2=c;
                 for mv = 1:length(mus)
                    mu = mus(mv)
             
            for  ei = 1:length(epsv)
                    e = epsv(ei)
                    avgerror = 0;
                    block_size = lengthA/(no_part*1.0);
                    part = 0;
                    t_1 = 0;
                    t_2 = 0;
                    while ceil((part+1) * block_size) <= lengthA
                   %% seprating testing and training datapoints for
                   % crossvalidation
                                t_1 = ceil(part*block_size);
                                t_2 = ceil((part+1)*block_size);
                                B_t = [A(t_1+1 :t_2,:)];
                                Data = [A(1:t_1,:); A(t_2+1:lengthA,:)];
%                                 [m_Data,n_Data]=size(Data);
%                                 sz=ceil(uvs1*(m_Data));
%                                 U1=U(1:sz,:);
                   %% testing and training
                                [accuracy(i),time] = NonLin_AULSTSVM(Data,B_t,U,c,c2,a,e,mu);
%                                 avgerror = avgerror + accuracy_with_zero;
                                part = part+1;
                                i=i+1;
                     end
%          time = mean(time1);
              acc = mean(accuracy);
              acc_SD = std(accuracy);
           %% updating optimum c, L
       
%           
               min_c1 = c;
               min_c2=c2;
                min_a = a;
               min_e = e;
%             min_u = u_per;
                min_u=u_per;
                min_mu=mu;
%            end 
            end
%               end
        end
        end

   end
 end

 fprintf(file1,'%s\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\tu=%g\tn=%g\n',file,size(A,1),size(A_test,1),acc,acc_SD,min_c1,min_c2,min_a,min_e,min_mu,time,min_u,n-1); 
 
end
 