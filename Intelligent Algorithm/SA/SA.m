clear;
clc;
%% preparing data set
filename = 'kroA200.tsp';
url = 'http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/kroA200.tsp';
urlwrite(url,filename);
file = fopen(filename);
flag = false;
coordinates = [];
while ~feof(file)
  str = fgetl(file);
  if strfind(str,'EOF')
    break
  elseif strfind(str,'NODE_COORD_SECTION')
    flag = true;
  elseif flag
    a = strsplit(str);
    index = str2num(char(a(1)));
    in_x = str2num(char(a(2)));
    in_y = str2num(char(a(3)));
    if ~coordinates
      coordinates = [in_x,in_y];
    end
    coordinates = [coordinates;[in_x,in_y]];
  else
    continue
  end
end
%% preparing the value numbers
alpha = 0.99;
T0 = 100;
Tf = 3;
t = T0;
%% 将城市打印在图上
x = coordinates(:,1);
y = coordinates(:,2);
figure;
set (gcf,'position',[20,50,1500,800]);
scatter(x,y,'r');
grid();
%% 计算城市之间的距离矩阵
while t>Tf
    break;
end
