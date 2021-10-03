function createfigure(xdata1, ydata1, zdata1, xdata2, ydata2, zdata2, xdata3, ydata3, zdata3)
%CREATEFIGURE(xdata1, ydata1, zdata1, xdata2, ydata2, zdata2, xdata3, ydata3, zdata3)
%  XDATA1:  surface xdata
%  YDATA1:  surface ydata
%  ZDATA1:  surface zdata
%  XDATA2:  surface xdata
%  YDATA2:  surface ydata
%  ZDATA2:  surface zdata
%  XDATA3:  surface xdata
%  YDATA3:  surface ydata
%  ZDATA3:  surface zdata

%  由 MATLAB 于 12-Jul-2021 19:36:13 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1,...
    'Position',[0.13 0.11 0.164353680224504 0.82343394072784]);
hold(axes1,'on');

% 创建 mesh
mesh(xdata1,ydata1,zdata1,'Parent',axes1);

% 创建 zlabel
zlabel('周期T','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 ylabel
ylabel('倔强系数k','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 xlabel
xlabel('半径r','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 title
title('周期T与半径r、倔强系数k的关系','FontWeight','bold','FontName','SimSun');

view(axes1,[-37.5 30]);
grid(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'GridLineStyle','--','LineWidth',1.5,'XMinorTick','on',...
    'YMinorTick','on');
% 创建 colorbar
colorbar(axes1);

% 创建 axes
axes2 = axes('Parent',figure1,...
    'Position',[0.410797101449275 0.11 0.17429769744393 0.814425293542481]);
hold(axes2,'on');

% 创建 mesh
mesh(xdata2,ydata2,zdata2,'Parent',axes2);

% 创建 zlabel
zlabel('周期T','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 ylabel
ylabel('质量m','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 xlabel
xlabel('半径r','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 title
title('周期T与半径r、质量m的关系','FontWeight','bold','FontName','SimSun');

view(axes2,[68.2736842105263 20.5398406374502]);
grid(axes2,'on');
hold(axes2,'off');
% 设置其余坐标区属性
set(axes2,'GridLineStyle','--','LineWidth',1.5,'XMinorTick','on',...
    'YMinorTick','on');
% 创建 colorbar
colorbar(axes2);

% 创建 axes
axes3 = axes('Parent',figure1,...
    'Position',[0.691594202898551 0.11 0.156434195152532 0.814424147987747]);
hold(axes3,'on');

% 创建 mesh
mesh(xdata3,ydata3,zdata3,'Parent',axes3);

% 创建 zlabel
zlabel('周期T','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 ylabel
ylabel('质量m','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 xlabel
xlabel('倔强系数k','FontWeight','bold','FontSize',7,'FontName','SimSun');

% 创建 title
title('周期T与倔强系数k、质量m的关系','FontWeight','bold','FontName','SimSun');

view(axes3,[-57.1 16.1]);
grid(axes3,'on');
hold(axes3,'off');
% 设置其余坐标区属性
set(axes3,'GridLineStyle','--','LineWidth',1.5,'XMinorTick','on',...
    'YMinorTick','on');
% 创建 colorbar
colorbar(axes3);

