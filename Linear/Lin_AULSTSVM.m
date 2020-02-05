function [accuracy,train_Time]=Lin_LSTWSVM(A,A_test,U,c,c2,a,e)
% c2=a;
[m,n]=size(A);[m_test,n_test]=size(A_test);
x0=A(:,1:n-1);y0=A(:,n);
xtest0=A_test(:,1:n_test-1);ytest0=A_test(:,n_test);
%----------------Training-------------
A=x0(find(y0(:,1)>0),:);B=x0(find(y0(:,1)<=0),:);
C=[A;B];
m1=size(A,1);m2=size(B,1);
m3=size(U,1);
e1=ones(m1,1);e2=ones(m2,1);
eu=ones(m3,1);
% K = zeros (m1,m3);
c1=c*c2;
tic
% K=B*C';
% for i =1: m1
%     for j =1: m3
%         u=[A(i ,:) 1];v=[C(j ,:) 1];
%         nom = norm( u - v );
%         K(i,j)=nom*nom;
%     end
% end
G=[B e2];
GTG=G'*G;
% GTG=GTG+(c2*speye(size(GTG,1)));
% invGTG=inv(GTG +(1e-5*speye(size(GTG,1))));
% K = zeros (m2,m3);
% K=U*C';

O=[U eu];
OTO=O'*O;
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
H=[A e1];
HTH=H'*H;
% HTH=HTH+(c2*speye(size(HTH,1)));
I=speye(size(HTH,1));
% Previous best technique
% alpha=(OTO+c*GTG+c*HTH+c2*I)\(c*e*H'*e1-c*e*G'*e2);
% alpha=(OTO+c2*I)\(H'*c1*e1-G'*c1*e2);
%%%%%%Good one%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp=(H'*e1-G'*e2);
u1=(OTO+c2*I)\(c1*temp);
% u1=(OTO+c*GTG+c*HTH+c2*I)\(H'*c*e1-G'*c*e2);
u2=(GTG+HTH+c2/c*I)\(temp+a/c*u1);
% u1=-(1/c*HTH+c2/c*I)\(-a/c*alpha);
% u2=(1/c*GTG+c2/c*I)\(a/c*alpha);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_Time=toc;
%---------------Testing---------------
no_test=size(xtest0,1);
% K=xtest0*C';
% K = zeros(no_test,m3);
% for i =1: no_test
%     for j =1: m3
%         u=xtest0(i ,:);v=C(j ,:);
%         nom = norm( u - v );
%         K(i,j )=nom*nom;
%     end
% end
K=[xtest0 ones(no_test,1)];
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
