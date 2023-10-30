%read hd5F files with extensions h5

clc
clear
close all

main_dir = './example';

path_dir = [main_dir, '/lpf/restart'];
data_dir = [main_dir, '/data'];
enkf_dir = [main_dir, '/enkf'];
etkf_dir = [main_dir, '/etkf'];
etkf_sqrt_dir = [main_dir, '/etkf_sqrt'];

nsimul = 20;
T = 10;
N = 100;
M = 1000;
L = 1;
coord = 1;
d = 5;
dim = d+2;
dim2 = dim^2;
dimx = 3*dim2;
path_h = zeros(dimx, N*(T+1), nsimul);
E_EnKF_h = zeros(dimx, T+1, nsimul);
X_sig = zeros(dimx,T+1);

filename = [data_dir, '/data.h5'];
for t = 0: T
    X_sig(:,t+1) = h5read(filename,sprintf('/step_%.8d/signal',t))';
end


for h = 0:nsimul -1 
    filename = [path_dir, sprintf('/restart_%.8d.h5',h)];
    path_h(:,:,h+1) = h5read(filename,'/path')';
    filename = [enkf_dir, sprintf('/sim_%.8d.h5',h)];
    E_EnKF_h(:,:,h+1) = h5read(filename,'/E_loc')';
end

filename = [path_dir, sprintf('/restart_%.8d.h5',nsimul-1)];
ESS_saved = h5read(filename,'/ess_saved')';


E_EnKF = mean(E_EnKF_h,3);
E_EnKF1 = squeeze(E_EnKF_h(coord,:,:));


path0 = mean(path_h,3);
E_LaggedPF = zeros(dimx,T+1);
for i = 1:T+1
    i1 = N*(i-1)+1;
    i2 = N*i;
    E_LaggedPF(:,i) = sum(path0(:,i1:i2),2)/N;
end

E_LaggedPF1 = zeros(T+1,nsimul);
for i = 1:T+1
    i1 = N*(i-1)+1;
    i2 = N*i;
    E_LaggedPF1(i,:) = sum(path_h(coord,i1:i2,:),2)/N;
end
%%
H_EnKF = zeros(dim,dim,T+1);
H_signal = zeros(dim,dim,T+1);
H_LPF = zeros(dim,dim,T+1);
for t = 1:T+1
    H_EnKF(:,:,t) = reshape(E_EnKF(1:dim2,t),dim,dim);
    H_signal(:,:,t) = reshape(X_sig(1:dim2,t),dim,dim);
    H_LPF(:,:,t) = reshape(E_LaggedPF(1:dim2,t),dim,dim);
end

min_v = 0;
max_v = 2.5;

%% errors
rel_err_EnKF = abs(E_EnKF - X_sig)./max(abs(E_EnKF),abs(X_sig));
max_rel_err_EnKF = max(rel_err_EnKF,[],'all');

rel_err_LPF = abs(E_LaggedPF - X_sig)./max(abs(E_LaggedPF),abs(X_sig));
max_rel_err_LPF = max(rel_err_LPF,[],'all');

max_rel_err = max(max_rel_err_LPF,max_rel_err_EnKF);

L2_X_sig = norm(X_sig);
L2_err_LPF = norm(E_LaggedPF - X_sig)/L2_X_sig;
L2_err_EnKF = norm(E_EnKF - X_sig)/L2_X_sig;

L2_X_sig1 = norm(X_sig(coord,:));
L2_err_LPF1 = norm(rel_err_LPF(coord,:))/L2_X_sig1;
L2_err_EnKF1 = norm(rel_err_EnKF(coord,:))/L2_X_sig1;
%%
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

figure
for t=1:T
    surf(H_EnKF(:,:,t),'CDataMapping','scaled')
    colormap(parula)
    title('H', 'FontSize', 30)
    axx = gca;
    zlim([0,2.5])
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 20;
    axx.YAxis.FontSize = 20;
    
    drawnow
end


%%
figure
titstr = {'Lagged PF','EnKF'};
set(gcf, 'Position',  [100, 100, 1200, 1200])
    
for i = 1 : 2
    ax(i) = subplot(2,1,i);
    if i == 1
        h1 = histogram(rel_err_LPF);
    elseif i == 2
        h1 = histogram(rel_err_EnKF);
    end
    h1.Normalization = 'probability';
    h1.BinWidth = 0.1;
    xlim([0, max_rel_err-0.5]);
    xticks(0:0.1:max_rel_err-0.5)
    ylim([0 0.7])
    yticks(0:0.05:1)
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 18;
    axx.YAxis.FontSize = 18;
    title(titstr{i}, 'FontSize', 34)
    xlabel('Relative Absolute Error', 'FontSize', 28)
    ylabel('Relative Probability',  'FontSize', 28)
end


%% Plot errors

map = [1 1 1 %white
        0 1 0 %green
       1 0 0 %red
       0 0 1 %blue
       ];

figure
axis_num = 1;
n_subplots = 2;
set(gcf, 'Position',  [100, 100, 1000, 1600])

ax(axis_num) = subplot(n_subplots,1,axis_num);
image(rel_err_LPF,'CDataMapping','scaled')
caxis ([0,max_rel_err])
colormap(ax(axis_num),map);%bluewhitered(3))
colorbar 
str  = sprintf('Relative Absolute Error for Lagged PF - (Relative $l_2$ error $= %.0e$)',L2_err_LPF);
title(str, 'FontSize', 30)
axx = gca;
axx.YDir = 'normal';
axx.XAxis.FontSize = 30;
axx.YAxis.FontSize = 30;
axis_num = axis_num + 1;

ax(axis_num) = subplot(n_subplots,1,axis_num);
image(rel_err_EnKF,'CDataMapping','scaled')
caxis ([0,max_rel_err])
colormap(ax(axis_num),map);%bluewhitered(3))
colorbar
str  = sprintf('Relative Absolute Error for EnKF - (Relative $l_2$ error $= %.0e$)',L2_err_EnKF);
title(str, 'FontSize', 30)
axx = gca;
axx.YDir = 'normal';
axx.XAxis.FontSize = 30;
axx.YAxis.FontSize = 30;



%% Plot for coord
time=0:T;

figure
set(gcf, 'Position',  [100, 100, 1200, 1600])

subplot(2,1,1)
for i = 1:nsimul-1
    hp = scatter(time,E_LaggedPF1(:,i),2,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_LaggedPF1(:,nsimul),2,'filled','MarkerFaceColor','g', 'DisplayName', 'LPF Simulations');
hold on
plot(time,X_sig(coord,:),'k-','DisplayName', 'Reference','LineWidth',2)
hold on
plot(time,E_LaggedPF(coord,:),'r-','DisplayName', 'Avg LPF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^%d$ w.r.t. the LPF distribution',coord);
str2 = sprintf('over %d simulations with: $L = %d$ and $N = %d$ ($L_2$-error $=%.2E$)',nsimul, L,N,L2_err_LPF1);
title({str1,str2},'FontSize', 30)
legend show
legend('Location','northeast', 'color','none')
set(legend, 'FontSize', 18, 'Orientation','horizontal')
ylim([-0.2+min(X_sig(coord,:)) 0.2+max(X_sig(coord,:))])
axx = gca;
axx.XAxis.FontSize = 18;
axx.YAxis.FontSize = 17;
xlabel('$n$', 'FontSize', 30)
ylabel('$E(X_n^1\,|data)$','FontSize', 30)

    
subplot(2,1,2)
for i = 1:nsimul-1
    hp = scatter(time,E_EnKF1(:,i),3,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_EnKF1(:,nsimul),3,'filled','MarkerFaceColor','g', 'DisplayName', 'EnKF Simulations');
hold on
plot(time,X_sig(coord,:),'k-','DisplayName', 'Reference','LineWidth',2)
hold on
plot(time,E_EnKF(coord,:),'r-','DisplayName', 'Avg EnKF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^%d$ w.r.t. the EnKF distribution',coord);
str2 = sprintf('over %d simulations with %d ensembles ($L_2$-error $=%.2E$)',nsimul, M, L2_err_EnKF1);
title({str1,str2},'FontSize', 30)
legend show
legend('Location','northeast', 'color','none')
set(legend, 'FontSize', 18, 'Orientation','horizontal')
ylim([-0.2+min(X_sig(coord,:)) 0.2+max(X_sig(coord,:))])
axx = gca;
axx.XAxis.FontSize = 18;
axx.YAxis.FontSize = 17;
xlabel('$n$', 'FontSize', 30)
ylabel('$E(X_n^1\,|data)$','FontSize', 30)


%% plot ESS
[mm,ind] = min(ESS_saved);
ESS_withoutZeros = ESS_saved(1:ind-1);
figure
set(gcf, 'Position',  [100, 100, 1200, 400])
plot(1:ind-1,ESS_withoutZeros,'k-', 'MarkerSize',10,'MarkerFaceColor','k')
str = sprintf('ESS up to T = %d', T);
title(str,'FontSize',30)
