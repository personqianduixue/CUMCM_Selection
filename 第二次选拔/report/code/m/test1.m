clc,warning off
% load Data_sheet.mat
mulu = readtable('二十支股票重要参数.xlsx','VariableNamingRule','preserve' ,'Sheet',1);
name = table2cell(mulu(:,2));

Data1 = Data_sheet{1};
range = timerange("2020-03-26","2021-03-26");
%%
figure(1)
value = [Data1(range,2:4),Data1(range,1)];
% value.Properties.VariableNames = {'open','high','low','close'};
candle(value)
%%
figure(2)
price = Data1(range,1:4);
% value.Properties.VariableNames = {'open','high','low','close'};
vars = {{'open','high','low','close'}};
stackedplot(price,vars,'Title','走势分析','LineWidth', 2)
%%
open = Data1(range,1);
open = table2array(open)
plot(1:height(open),open)

beautiplot

% for i=1:20
%     i
%     Data1 = Data_sheet{i};
%     range = timerange("2020-03-26","2021-03-26");
%     value = Data1(range,1:4);
%     value.Properties.VariableNames = {'close','open','high','low'};
%     candle(value),hold on
% end
% legend(name)