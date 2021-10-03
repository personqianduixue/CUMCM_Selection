load theta_theta12_x1x2_q3.mat
theta_0 = theta_q3;
theta1_0 = theta1_q3;
theta2_0 = theta2_q3;
x1_0 = x1_q3;
x2_0 = x2_q3;
tspan = [0,100];
tt0 = [theta_0 0 x1_0 x2_0 theta1_0 theta2_0];
opts = odeset('RelTol',1e-2,'AbsTol',1e-4);
[t,theta] = ode45(@(t,theta) funq3_2_2(t,theta), tspan, tt0);
disp('ok')
%%
% plot(t,theta)
% legend('\theta','\theta^{\prime}','x1','x2','\theta_1','\theta_2')
figure('Position',[229.8,248.2,906.4,353.4])
leng = floor(length(t)/20);
plot(t(1:leng),theta(1:leng,1),'r-',t(1:leng),theta(1:leng,2),'b--');
legend('\theta','\theta^{\prime}')
xlabel('时间 t/s');ylabel('模型3-2 角度\theta及角速度\theta^{\prime}');
title('模型3-2:角度及角速度随时间的变化(5s)');
legend('\theta(t)','\theta^{\prime}(t)','location','northeast')
beautiplot('small')
exportgraphics(gcf,'img\模型3-2角度及角速度随时间的变化5s.png','Resolution',600)

%%
figure('Position',[97.8,395.4,979.2,279.2]);subplot(121)
plot(t, theta(:, 1), 'r-')
xlabel('时间 t/s');
ylabel('角度\theta(t)');
title('模型3-2:角度随时间(100s)的变化\theta(t)');
subplot(122);plot(t, theta(:, 2), 'b-')
xlabel('时间 t/s');
ylabel('角速度\theta^{\prime}(t)');
title('模型3-2:角速度随时间的变化(100s)\theta^{\prime}(t)');
beautiplot
exportgraphics(gcf, 'img\模型3-2角度及角速度随时间的变化100s.png', 'Resolution', 600)