function [accuracy,train_Time]=NonLin_AULSTSVM(A,A_test,U,c,c2,a,e,mu)
% c2=a;
mew1=1/(2*mu*mu);
[m,n]=size(A);[m_test,n_test]=size(A_test);
x0=A(:,1:n-1);y0=A(:,n);
xtest0=A_test(:,1:n_test-1);ytest0=A_test(:,n_test);
%----------------Training-------------
A=x0(find(y0(:,1)>0),:);B=x0(find(y0(:,1)<=0),:);
C=[A;B];
m1=size(A,1);m2=size(B,1);
m3=size(U,1);
m4=size(C,1);
e1=ones(m1,1);e2=ones(m2,1);
eu=ones(m3,1);

c1=c*c2;
tic
K = zeros (m1,m4);
for i =1: m1
    for j =1: m4
%         u=[A(i ,:)];v=[C(j ,:)];
        nom = norm( A(i ,:) - C(j ,:) );
        K(i,j)=exp(-mew1*nom*nom);
%         H1(i,j ) =exp(-mew*((u-v)*(u-v)'));
    end
end

H=[K e1];
% HTH=H'*H;

K = zeros (m2,m4);
for i =1: m2
    for j =1: m4
%         u=[B(i ,:)];v=[C(j ,:)];
        nom = norm( B(i ,:) - C(j ,:) );
        K(i,j)=exp(-mew1*nom*nom);
%         H2(i,j ) = exp(-mew*((u-v)*(u-v)'));
    end
end

G=[K e2];
% GTG=G'*G;


K = zeros (m3,m4);
for i =1: m3
    for j =1: m4
%         u=[B(i ,:)];v=[C(j ,:)];
        nom = norm( U(i ,:) - C(j ,:) );
        K(i,j)=exp(-mew1*nom*nom);
%         H2(i,j ) = exp(-mew*((u-v)*(u-v)'));
    end
end
O=[K eu];
% OTO=O'*O;
% GTG=GTG+(c2*speye(size(GTG,1)));
% invGTG=inv(GTG +(1e-5*speye(size(GTG,1))));
% K = zeros (m2,m3);
% K=A*C';

% for i =1: m2
%     for j =1: m3
%         u=[B(i ,:) 1];v=[C(j ,:) 1];
%         nom = norm( u - v );
%         K(i,j)=nom*nom;
%     end
% end

% HTH=HTH+(c2*speye(size(HTH,1)));
I1=speye(m1);
I2=speye(m2);
I3=speye(m3);
I=speye(m+1);

temp=(H'*e1-G'*e2);
% u1=(O'*O+c2*I)\(c1*temp);
u1=1/c2*(I-O'*((c2*I3+O*O')\O))*(c1*temp);
% u2=(G'*G+H'*H+c2/c*I)\(temp+a/c*u1);
if(m1>m2)
    invA=c/c2*(I-H'*((c2/c*I1+H*H')\H));
    u2=(invA-invA*G'*((I2+G*invA*G')\G)*invA)*(temp+a/c*u1);
else
    invA=c/c2*(I-G'*((c2/c*I2+G*G')\G));
    u2=(invA-invA*H'*((I1+H*invA*H')\H)*invA)*(temp+a/c*u1);
end
%  

% u2=(GTG+HTH+c2*I)\(c*(H'*e1-G'*e2)+a*u1);
% u1=-(1/c*HTH+c2/c*I)\(-a/c*alpha);
% u2=(1/c*GTG+c2/c*I)\(a/c*alpha);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_Time=toc;
%---------------Testing---------------
no_test=size(xtest0,1);
K = zeros(no_test,m4);
for i =1: no_test
    for j =1: m4
%         u=xtest0(i ,:);v=C(j ,:);
        nom = norm( xtest0(i ,:) - C(j ,:) );
        K(i,j )=exp(-mew1*nom*nom);
    end
end
K=[K ones(no_test,1)];
% K=[xtest0 ones(no_test,1)];
% preY1=K*u1/norm(u1(1:size(u1,1)-1,:));
preY2=K*u2/norm(u2(1:size(u2,1)-1,:));
predicted_class=[];
for i=1:no_test
     if preY2(i)>0
        predicted_class=[predicted_class;1];
    else
        predicted_class=[predicted_class;-1];
    end

end
err = sum(predicted_class ~= ytest0);
accuracy=(no_test-err)/(no_test)*100


return
end
