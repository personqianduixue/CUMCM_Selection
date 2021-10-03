figure('Position',[159.4,318.6,1280.8,268.8])
fontname = 'SimSun';
smallfont = 7;
largefont = 10;
subplot(131)
[rp,kp] = meshgrid(0:0.01:1,400:1:600);
m=5;
Tp = 2*pi./(sqrt(8*9.8./(3*pi*rp)+4*kp/m));

% surf(rp,kp,Tp)
mesh(rp,kp,Tp)
% colormap('cool');
colorbar;
xlabel('半径r','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
ylabel('倔强系数k','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
zlabel('周期T','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
title('周期T与半径r、倔强系数k的关系','FontSize',largefont,'FontWeight', 'bold','FontName',fontname);grid on
beautiplot3()
%%
% figure(2)
subplot(132)
[rp,mp] = meshgrid(0:0.01:1,0:0.5:50);
k=500;
Tp = 2*pi./(sqrt(8*9.8./(3*pi*rp)+4*k./mp));

% surf(rp,kp,Tp)
mesh(rp,mp,Tp)
% colormap('cool');
colorbar;
xlabel('半径r','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
ylabel('质量m','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
zlabel('周期T','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
title('周期T与半径r、质量m的关系','FontSize',largefont,'FontWeight', 'bold','FontName',fontname);
grid on
view(71.7,18.9)
beautiplot3()

%%
% figure(3)
subplot(133)
[kp,mp] = meshgrid(400:1:600,0:0.5:50);
r=0.3;
Tp = 2*pi./(sqrt(8*9.8./(3*pi*r)+4*kp./mp));

% surf(rp,kp,Tp)
mesh(kp,mp,Tp)
% colormap('cool');
colorbar;
view(-57.1,16.1)
xlabel('倔强系数k','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
ylabel('质量m','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
zlabel('周期T','FontSize',smallfont,'FontWeight', 'bold','FontName',fontname);
title('周期T与倔强系数k、质量m的关系','FontSize',largefont,'FontWeight', 'bold','FontName',fontname);
grid on
beautiplot3()


