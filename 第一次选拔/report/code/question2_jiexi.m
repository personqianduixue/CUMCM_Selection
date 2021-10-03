syms theta(t) omega(t) a b
load thetax1x2_q1.mat
eqn = diff(theta,t,2)==a*sin(theta)+b*sin(2*theta)/2;
Dtheta = diff(theta,t);
cond = [theta(0)==theta_q1, Dtheta(0)==0];
ySol(t) = dsolve(eqn,cond,'Implicit',true)
