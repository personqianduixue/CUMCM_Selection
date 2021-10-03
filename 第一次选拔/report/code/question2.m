clc, clear
load thetax1x2_q1.mat
theta_0 = roundn(double(theta_q1), -4);
tspan = [0, 1000];
tt0 = [theta_0, 0];
% opts = odeset('RelTol', 1e-2, 'AbsTol', 1e-4);
[t, theta] = ode45(@(t, theta) funq2(t, theta), tspan, tt0);
%%
figure('Position', [229.8, 248.2, 906.4, 353.4])
plot(t, theta(:, 1), 'r-', t, theta(:, 2), 'b--')
xlabel('时间 t/s');
ylabel('模型1角度\theta及角速度\theta^{\prime}');
title('模型1角度及角速度随时间的变化');
legend('\theta(t)', '\theta^{\prime}(t)', 'location', 'northeast')
beautiplot('small')
% exportgraphics(gcf, 'img\模型1角度及角速度随时间的变化.png', 'Resolution', 600)
%%
figure('Position',[97.8,395.4,979.2,279.2]);subplot(121)
plot(t, theta(:, 1), 'r-')
xlabel('时间 t/s');
ylabel('角度\theta(t)');
title('模型1角度随时间(1000s)的变化\theta(t)');
subplot(122);plot(t, theta(:, 2), 'b-')
xlabel('时间 t/s');
ylabel('角速度\theta^{\prime}(t)');
title('模型1角速度随时间(1000s)的变化\theta^{\prime}(t)');
beautiplot('small')
% exportgraphics(gcf, 'img\模型1角度及角速度随时间的变化1000s.png', 'Resolution', 600)
%%
% cftool(t,theta(:,2))
% h = hilbert(theta(:,1));
% [up,lo] = envelope(theta(:,1));
% figure;
% plot(t,theta(1),'b-',t,up,'r--')
%%
figure
alpha = diff(theta(:, 2))./diff(t);
plot3(theta(2:end, 1),theta(2:end, 2),alpha,'b-')
xlabel('角度\theta','FontSize',10,'FontWeight', 'bold');
ylabel('角速度\omega','FontSize',10,'FontWeight', 'bold');
zlabel('角加速度\alpha','FontSize',10,'FontWeight', 'bold');
title('角度、角速度、角加速度相图(1000s)','FontSize',12,'FontWeight', 'bold');grid on
view(-24.79,51.15)
beautiplot3()
exportgraphics(gcf, 'img\模型1角度角速度角加速度相图1000s.png', 'Resolution', 600)
%%
figure('Position', [326.6, 305.8, 556, 353.6])
plot(theta(:, 1), theta(:, 2), 'b-')
xlabel('角度\theta');
ylabel('角速度\theta^{\prime}');
title('角度 角速度相图');
beautiplot('small')
exportgraphics(gcf, 'img\模型1角度及角速度相图.png', 'Resolution', 600)

%%
x = theta(:, 1);
y = theta(:, 2);
%cftool(t, y)
a1 = 3.211;
b1 = 20.56;
c1 = 3.141;
% z = a1*sin(b1*t+c1);
z = a1*b1*cos(b1*t+c1);
plot3(x,y,z,'r','LineWidth',2)
xlabel('角度\theta','FontSize',10,'FontWeight', 'bold');
ylabel('角速度\omega','FontSize',10,'FontWeight', 'bold');
zlabel('角加速度\alpha','FontSize',10,'FontWeight', 'bold');
title('角度、角速度、角加速度相图','FontSize',12,'FontWeight', 'bold');grid on
beautiplot3()
% beautiplot('small')
exportgraphics(gcf, 'img\模型1角度角速度角加速度相图(5s).png', 'Resolution', 600)
