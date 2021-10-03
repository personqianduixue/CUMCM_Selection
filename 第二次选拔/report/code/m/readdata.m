clc,clear
Data_sheet = cell(1,20);
for i=1:20
Data_sheet{i} = readtable('二十支股票重要参数.xlsx','VariableNamingRule','preserve' ,'Sheet',i+1);
% Data_sheet = readtable('二十支股票重要参数.xlsx','VariableNamingRule','preserve' ,'Sheet',2);
summary(Data_sheet{i})
Data_sheet1 = Data_sheet{i};
timestr = table2cell(Data_sheet1(:,'时间'));
date = datetime(timestr,'InputFormat','yyyy-MM-dd');
other = Data_sheet1(:,2:end);
d = table(date);
all = [d,other];
% all(:,1)
all2 = table2timetable(all);
all2.Properties.VariableNames = {'close','open','high','low','zdiefu','zdiee'...
    ,'cjliang','cje','zhenfu','hsliu'};
writetimetable(all2,'all.xlsx','Sheet',i+1)
Data_sheet{i} = all2;
% range = timerange("2018-03-26","2021-03-26");
% all2(range,:)
end
%%
save Data_sheet.mat Data_sheet
%%

