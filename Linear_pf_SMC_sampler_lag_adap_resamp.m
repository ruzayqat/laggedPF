%In this file, in MCMC step from n=1:L, it updates the whole path

clc
close all
clear 
format long;

Model = 'Linear';
read_params_from_file = 0;
save_files = 0;

%% Create a folder for figures and ouput files
%first get the date of today and the time
date = floor(clock);
%create a folder  with name: Output-year-month-day-hours-minutes (e.g.
%create a folder  with name: Output-year-month-day-hours-minutes (e.g.
%Lorenz96_Output-2021-6-28-11-40)
OutputFolderName = sprintf([Model,'_Output-%d-%d-%d-%d-%d'], date(1:5));
mkdir(OutputFolderName)
%create three folders inside the Output folder: Data, Figures, SimulOutput
writematrix(date(1:5),[OutputFolderName,'/date.dat'])
DataFolderName = [OutputFolderName,'/Data'];
FiguresFolderName = [OutputFolderName,'/Figures'];
SimulOutputFolderName = [OutputFolderName,'/SimulOutput'];
RefOutputFolderName = [OutputFolderName,'/RefOutput'];
mkdir(DataFolderName)
mkdir(FiguresFolderName)
mkdir(SimulOutputFolderName)
mkdir(RefOutputFolderName)


%% Check if export_fig-master folder is in this current folder,
% otherwise download it
% export_fig function is used to generate high quality plots in pdf or
% other formats
if ~exist('export_fig-master', 'dir')
    url = 'https://github.com/altmany/export_fig/archive/refs/heads/master.zip';
    outfilename = websave([pwd,'/export_fig-master'],url);
    unzip('export_fig-master.zip')
end
addpath([pwd,'/export_fig-master'])
%% Set some parameters
t1 = tic; 

dimx = 20;

nproc = 8;
multipleOfnproc = 4;
nsimul = multipleOfnproc * nproc; 


o_freq_spatial = 1; 
o_freq_time = 1;

dimo = floor(dimx / o_freq_spatial); % dimension of observations
if dimo == dimx
    C_is_eye = 1; %Is C an identity matrix?
else
    C_is_eye = 0;
end

N = 100; %number of  particles 
M = 100; %number of ensembles

L = 1; %Lag

theta = 0.5;

MCMC_steps_max = 100;
MCMC_steps_min = 15;

avg_accep_rate_min = 0.1700;
avg_accep_rate_max = 0.3000;

bisection_nmax = 1000;

ESS_min = 0.8*N;

phi1 = 0.001;

T = 100; %number of iterations

x_star = 1.5 * ones(dimx,1);

zero_vec_dx = zeros(dimx,1);
I_dx = eye(dimx);

zero_vec_dy = zeros(dimo,1);
I_dy = eye(dimo);

log2pi = log(2*pi);

sig_x = sqrt(1); %R1 = (sig_x * eye(dx)) * (sig_x * eye(dx)) ;
sig_y = sqrt(2);

%temp = 0.5*rand(dx,1);
%sig = diag(temp);

sig = 2.38^2/dimx;



%% Choose the matrices A, R_1, R_2 and C
%here we use A = theta *eye(dimx) ... therfore we don't write A
if C_is_eye %only when dimo = dimx
    C = 1;
else
    %use this C or give another C
    C = sparse(dimo, dimx);
    if o_freq_spatial == 0 || o_freq_spatial >= dimx
        error('o_freq must be in ]0,dimx[')
    end
    j = 1; % always observe  the first coordinate
    for i = 1:dimo
        C(i, j) = 1;
        j = j + o_freq_spatial;
    end
end

%e = ones(dy,1);


R2_sqrt = sig_y * speye(dimo); %(spdiags([e 2*e e],-1:1,dy,dy))*20; %sqrt(80)* eye(dy);
R2 = sig_y^2  * speye(dimo); %R2_sqrt * R2_sqrt;
R2_inv = speye(dimo)/sig_y^2 ;%Identity_dy/R2; %covariance of g(yn|xn);

R2_sqrt_inv  = speye(dimo)/sig_y ;

ldet_R2 = 2*dimo*log(sig_y);%log(det(R2)) = log(sig_y^(2*dy))=2*dy * log(sig_y)

R1_sqrt = sig_x * speye(dimx);
R1 = sig_x^2 *speye(dimx);

ldet_R1 = 2*dimx*log(sig_x); %= log(det(R1));
%% Generate the data 
Y = zeros(dimo, T);
X_sig = zeros(dimx,T+1);
X_sig(:,1) = x_star;

for n = 1: T
    dW = randn(dimx,1);
    X_sig(:,n+1) = theta * X_sig(:,n) + sig_x * dW;
    if mod(n,o_freq_time) == 0
        dV = randn(dimo,1);
        Y(:,n) = C*X_sig(:,n+1) + sig_y * dV;
    end

end

vars = {'X', 'dW', 'dV','zero_vec_dx','I_dx','zero_vec_dy','I_dy'};
clear(vars{:});


writematrix(Y,[DataFolderName,'/','observations.dat'])

fprintf('finished generating the observations\n')


%% run KF
[E_truth, X_f, P_f] = KF(dimx,T,theta,R1,R2,C,Y, x_star,o_freq_time);
P_f = ndSparse(P_f);

% Write E_truth to a file
hFilenamePath = [RefOutputFolderName,'/E_truth.dat'];
writematrix(E_truth,hFilenamePath)
%% Run PF_with_SMC_Sampler nsimul times 

parfor h = 1:nsimul

    if h ~= nsimul
        [path0, ~] = LaggedPF(h,log2pi,L,dimx,dimo,theta,N,T,Y,x_star,phi1,...
        bisection_nmax,sig_x,sig_y,ldet_R1, ldet_R2,...
        ESS_min, MCMC_steps_min,MCMC_steps_max, avg_accep_rate_min,...
        avg_accep_rate_max, sig, X_f, P_f,C,o_freq_time);
    else
        [path0, ESS_saved0] = LaggedPF(h,log2pi,L,dimx,dimo,theta,N,T,Y,x_star,phi1,...
        bisection_nmax,sig_x,sig_y,ldet_R1, ldet_R2,...
        ESS_min, MCMC_steps_min,MCMC_steps_max, avg_accep_rate_min,...
        avg_accep_rate_max, sig, X_f, P_f,C,o_freq_time);
    end

    %ESS_saved_h{h} = ESS_saved0;
    %path_h(:,:,h) = path0;
    hFilenamePath = sprintf([SimulOutputFolderName,'/SimulPath_%d.dat'], h);
    if h == nsimul
        hFilenameESS = [SimulOutputFolderName,'/ESS.dat'];
        writematrix(ESS_saved0,hFilenameESS)
    end
    writematrix(path0,hFilenamePath)
end

SimulationTime = toc(t1);
fprintf('SimulationTime = %.4f\n',SimulationTime)
%%
% expecation of varphi(x) = x, x in R^dx w.r.t. to the filter using the last updated
% path
path_h = zeros(dimx,N*(T+1),nsimul);
hFilenameESS = [SimulOutputFolderName,'/ESS.dat'];
ESS_saved = readmatrix(hFilenameESS);
parfor h = 1:nsimul
    hFilenamePath = sprintf([SimulOutputFolderName,'/SimulPath_%d.dat'], h);
    path_h(:,:,h) = importdata(hFilenamePath);
    fprintf('Done reading %s data\n',hFilenamePath)
end


path0 = sum(path_h,3)/nsimul;
E_LaggedPF = zeros(dimx,T+1);
for i = 1:T+1
    i1 = N*(i-1)+1;
    i2 = N*i;
    E_LaggedPF(:,i) = sum(path0(:,i1:i2),2)/N;
end
%%
E_LaggedPF1 = zeros(T+1,nsimul);
for i = 1:T+1
    i1 = N*(i-1)+1;
    i2 = N*i;
    E_LaggedPF1(i,:) = sum(path_h(1,i1:i2,:),2)/N;
end

er2 = zeros(T+1,nsimul);
mse = zeros(T+1,1);
for i = 1:T+1
    er2(i,:) = (E_LaggedPF1(i,:) - E_truth(1,i)).^2;
    mse(i) = mean(er2(i,:));
end
%% EnKF

E_h = zeros(dimx,T+1,nsimul);
time_simul = zeros(nsimul,1);
parfor h = 1 : nsimul
    E_loc = EnKF(M,dimx,dimo,theta,T,x_star,Y,C,sig_x,R2,R2_sqrt, o_freq_time);
    E_h(:,:,h) = E_loc;
end
fprintf('Done with EnKF\n')
E_EnKF = mean(E_h,3);

E_EnKF1 = zeros(T+1,nsimul);
for i = 1:T+1
    E_EnKF1(i,:) = E_h(1,i,:);
end
%%%
% See the book "DATA ASSIMILATION IN MAGNETOHYDRODYNAMICS SYSTEMS USING
% KALMAN FILTERING " page 105

%% SQRT-ETKF
parfor h = 1 : nsimul
    E_loc= EtKF_sqrt(M,dimx,theta,T,x_star,Y,C,sig_x, R2, o_freq_time);
    E_h(:,:,h) = E_loc;
end
fprintf('Done with SQRT-ETKF\n')
E_Sqrt_ETKF = mean(E_h,3);

E_Sqrt_ETKF1 = zeros(T+1,nsimul);
for i = 1:T+1
    E_Sqrt_ETKF1(i,:) = E_h(1,i,:);
end

%% EtKF
parfor h = 1 : nsimul
   E_loc= EtKF(M,dimx,dimo,theta,T,x_star,Y,C,sig_x, R2_sqrt_inv, o_freq_time);
    E_h(:,:,h) = E_loc;
end
fprintf('Done with ETKF\n')
E_ETKF = mean(E_h,3);

E_ETKF1 = zeros(T+1,nsimul);
for i = 1:T+1
    E_ETKF1(i,:) = E_h(1,i,:);
end
%% write E_laggedPF, E_LaggedPF1 and E_KF to files
hFilenamePath = [SimulOutputFolderName,'/E_LaggedPF.dat'];
writematrix(E_LaggedPF,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_LaggedPF1.dat'];
writematrix(E_LaggedPF1,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_KF.dat'];
writematrix(E_truth,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_EnKF.dat'];
writematrix(E_EnKF,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_EnKF1.dat'];
writematrix(E_EnKF1,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_ETKF.dat'];
writematrix(E_ETKF,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_ETKF1.dat'];
writematrix(E_ETKF1,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_SQRT_ETKF.dat'];
writematrix(E_Sqrt_ETKF,hFilenamePath)
hFilenamePath = [SimulOutputFolderName,'/E_SQRT_ETKF1.dat'];
writematrix(E_Sqrt_ETKF1,hFilenamePath)
%%
err_LPF = abs(E_LaggedPF - E_truth);
rel_err_LPF = err_LPF ./ max(abs(E_truth),abs(E_LaggedPF));

err_EnKF = abs(E_EnKF - E_truth);
rel_err_EnKF = err_EnKF ./ max(abs(E_truth), abs(E_EnKF) );

err_ETKF = abs(E_ETKF - E_truth);
rel_err_ETKF = err_ETKF ./ max(abs(E_truth), abs(E_ETKF) );

err_ETKF_sqrt = abs(E_Sqrt_ETKF - E_truth);
rel_err_ETKF_sqrt = err_ETKF_sqrt ./ max(abs(E_truth), abs(E_Sqrt_ETKF));

max_v = max(max([E_LaggedPF(:),E_truth(:), E_EnKF(:), E_ETKF(:), E_Sqrt_ETKF(:)]));
min_v = min(min([E_LaggedPF(:),E_truth(:), E_EnKF(:), E_ETKF(:), E_Sqrt_ETKF(:)]));

min_E_truth = min(min(E_truth(:)));
max_E_truth = max(max(E_truth(:)));

max_LPF = max(max(E_LaggedPF(:)));
min_LPF = min(min(E_LaggedPF(:)));


max_err_pf = max(max(err_LPF(:)));
max_err_EnKF = max(max(err_EnKF(:)));
max_err_ETKF = max(max(err_ETKF(:)));
max_err_ETKF_sqrt = max(max(err_ETKF_sqrt(:)));

max_err = max([max_err_pf,max_err_EnKF,max_err_ETKF,max_err_ETKF_sqrt]);

L2_E_truth1 = norm(E_truth(1,:));
L2_err_LPF1 = norm(err_LPF(1,:))/L2_E_truth1;
L2_err_EnKF1 = norm(err_EnKF(1,:))/L2_E_truth1;
L2_err_ETKF1 = norm(err_ETKF(1,:))/L2_E_truth1;
L2_err_ETKF_sqrt1 = norm(err_ETKF_sqrt(1,:))/L2_E_truth1;

L2_E_truth = norm(E_truth);
L2_err_LPF = norm(err_LPF)/L2_E_truth;
L2_err_EnKF = norm(err_EnKF)/L2_E_truth;
L2_err_ETKF = norm(err_ETKF)/L2_E_truth;
L2_err_ETKF_sqrt = norm(err_ETKF_sqrt)/L2_E_truth;

max_rel_err =  max(max([rel_err_LPF(:),rel_err_EnKF(:),rel_err_ETKF(:), rel_err_ETKF_sqrt(:)]));
%% plot EnKF and ETKF and the absolute errors
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])
 
%returns a six-element date vector containing the current date and time in decimal form:
%[year month day hour minute seconds]

number_of_slices = 6;
figure
n_subplots = 5;
axis_num = 1;
set(gcf, 'Position',  [100, 100, 1000, 1500])

    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(E_truth,'CDataMapping','scaled')
    caxis ([min_v,max_v])
    colormap(ax(axis_num),parula)
    c1 = colorbar;
    c1.FontSize = 12;
    title('Reference (Kalman Filter)', 'FontSize', 20)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    %ylim([269,271])
    %xlim([167,169])
    axis_num = axis_num + 1;
    
    
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(E_LaggedPF,'CDataMapping','scaled')
    caxis ([min_v,max_v])
    colormap(ax(axis_num),parula)
    c1 = colorbar;
    c1.FontSize = 12;
    str = sprintf('Lagged PF (Average of %d simulations, $L = %d$ and $N = %d$ particles)',nsimul,L,N);
    title(str, 'FontSize', 16)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    %ylim([269,271])
    %xlim([167,169])
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(E_EnKF,'CDataMapping','scaled')
    caxis ([min_v,max_v])
    colormap(ax(axis_num),parula)
    c1 = colorbar;
    c1.FontSize = 12;
    str = sprintf('EnKF (Average of %d simulations with %d ensembles)',nsimul,M);
    title(str, 'FontSize', 18)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    %ylim([269,271])
    %xlim([167,168])
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(E_ETKF,'CDataMapping','scaled')
    caxis ([min_v,max_v])
    colormap(ax(axis_num),parula)
    c1 = colorbar;
    c1.FontSize = 12;
    str = sprintf('ETKF (Average of %d simulations with %d ensembles)',nsimul,M);
    title(str, 'FontSize', 18)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    %ylim([269,271])
    %xlim([167,169])
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(E_Sqrt_ETKF,'CDataMapping','scaled')
    caxis ([min_v,max_v])
    colormap(ax(axis_num),parula)
    c1 = colorbar;
    c1.FontSize = 12;
    str = sprintf('SQRT-ETKF (Average of %d simulations with %d ensembles)',nsimul,M);
    title(str, 'FontSize', 18)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    %ylim([269,271])
    %xlim([167,169])
    
    if save_files
        export_fig(sprintf([FiguresFolderName,'/',Model,...
        '_phi=X_dx=%d_dy=%d_N=%d_L=%d_T%d_freq_time=%d_freq_spatial=%d.pdf'],...
        dimx,dimo,N, L,T, o_freq_time, o_freq_spatial),'-m3')
    end
%% 
n_subplots = 4;
axis_num = 1;
map = [1 1 1 %white
        0 1 0 %green
       1 0 0 %red
       0 0 1 %blue
       ];
       %1 0 1 %magenta
       %0 0 0 
       %0 0 0
       %0 0 0
       %0 0 0]; %black
   
figure
set(gcf, 'Position',  [400, 100, 1000, 1500]) 

    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(rel_err_LPF,'CDataMapping','scaled')
    caxis ([0,max_rel_err])
    colormap(ax(axis_num),map)
    c1 = colorbar;
    c1.FontSize = 14;
    titstr = sprintf('Relative Absolute Error for Lagged PF-(Relative $L_2$ error $= %.1e$)',L2_err_LPF);
    title(titstr, 'FontSize', 16)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(rel_err_EnKF,'CDataMapping','scaled')
    caxis ([0,max_rel_err])
    colormap(ax(axis_num),map)
    c1 = colorbar;
    c1.FontSize = 14;
    titstr = sprintf('Relative Absolute Error for EnKF-(Relative $L_2$ error $= %.1e$)',L2_err_EnKF);
    title(titstr, 'FontSize', 16)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(rel_err_ETKF,'CDataMapping','scaled')
    caxis ([0,max_rel_err])
    colormap(ax(axis_num),map)
    c1 = colorbar;
    c1.FontSize = 14;
    titstr = sprintf('Relative Absolute Error for ETKF-(Relative $L_2$ error $= %.1e$)',L2_err_ETKF);
    title(titstr, 'FontSize', 16)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots,1,axis_num);
    image(rel_err_ETKF_sqrt,'CDataMapping','scaled')
    caxis ([0,max_rel_err])
    colormap(ax(axis_num),map)
    c1 = colorbar;
    c1.FontSize = 14;
    titstr = sprintf('Relative Absolute Error for SQRT-ETKF-(Relative $L_2$ error $= %.1e$)',L2_err_ETKF_sqrt);
    title(titstr, 'FontSize', 16)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;

    if save_files
        export_fig(sprintf([FiguresFolderName,'/',Model,...
        '_absolute_errors_phi=X_dx=%d_dy=%d_N=%d_L=%d_T%d_freq_time=%d_freq_spatial=%d.pdf'],...
        dimx,dimo,N, L,T, o_freq_time, o_freq_spatial),'-m3')
    end
    
%%    
figure
n_subplots = 4;
axis_num = 1;
set(gcf, 'Position',  [100, 100, 1200, 1200])
    
    ax(axis_num) = subplot(2,2,axis_num);
    h1 = histogram(rel_err_LPF);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.025;
    axis_num = axis_num + 1;
    xlim([0, max_rel_err]);
    xticks(0:0.05:max_rel_err)
    ylim([0 1])
    yticks(0:0.05:1)
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 12;
    axx.YAxis.FontSize = 12;
    titstr = 'Lagged PF';
    title(titstr, 'FontSize', 18)
    xlabel('Relative Absolute Error', 'FontSize', 18)
    ylabel('Relative Probability',  'FontSize', 18)
    
    
    
    ax(axis_num) = subplot(2,2,axis_num);
    h1 = histogram(rel_err_EnKF);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.025;
    axis_num = axis_num + 1;
    xlim([0, max_rel_err]);
    xticks(0:0.05:max_rel_err)
    ylim([0 1])
    yticks(0:0.05:1)
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 12;
    axx.YAxis.FontSize = 12;
    titstr = 'EnKF';
    title(titstr, 'FontSize', 18)
    xlabel('Relative Absolute Error', 'FontSize', 18)
    ylabel('Relative Probability',  'FontSize', 18)
    
    
    ax(axis_num) = subplot(2,2,axis_num);
    h1 = histogram(rel_err_ETKF);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.025;
    axis_num = axis_num + 1;
    xlim([0, max_rel_err]);
    xticks(0:0.05:max_rel_err)
    ylim([0 1])
    yticks(0:0.05:1)
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 12;
    axx.YAxis.FontSize = 12;
    titstr = 'ETKF';
    title(titstr, 'FontSize', 18)
    xlabel('Relative Absolute Error', 'FontSize', 18)
    ylabel('Relative Probability',  'FontSize', 18)
    
    
    ax(axis_num) = subplot(2,2,axis_num);
    h1 = histogram(rel_err_ETKF_sqrt);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.025;
    axis_num = axis_num + 1;
    xlim([0, max_rel_err]);
    xticks(0:0.05:max_rel_err)
    ylim([0 1])
    yticks(0:0.05:1)
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 12;
    axx.YAxis.FontSize = 12;
    titstr = 'ETKF-SQRT';
    title(titstr, 'FontSize', 18)
    xlabel('Relative Absolute Error', 'FontSize', 18)
    ylabel('Relative Probability',  'FontSize', 18)
    
    
    if save_files
        export_fig(sprintf([FiguresFolderName,'/',Model,...
        '_errors_Histo_phi=X_dx=%d_dy=%d_N=%d_L=%d_T%d_freq_time=%d_freq_spatial=%d.pdf'],...
        dimx,dimo,N, L,T, o_freq_time, o_freq_spatial),'-m3')
    end
%% Plot E1 vs E_KF
time=0:T;
figure
set(gcf, 'Position',  [100, 100, 1200, 1600])

subplot(4,1,1)
for i = 1:nsimul-1
    hp = scatter(time,E_LaggedPF1(:,i),3,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_LaggedPF1(:,nsimul),3,'filled','MarkerFaceColor','g', 'DisplayName', 'LPF Simulations');
hold on
plot(time,E_truth(1,:),'k-','DisplayName', 'Reference','LineWidth',2)
hold on
plot(time,E_LaggedPF(1,:),'r-','DisplayName', 'Avg LPF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^1$ w.r.t. the LPF distribution');
str2 = sprintf('over %d simulations with: $L = %d$ and $N = %d$ ($L_2$-error $=%.2E$)',nsimul, L,N,L2_err_LPF1);
title({str1,str2},'FontSize', 16)
legend show
legend('Location','northeast')
set(legend, 'FontSize', 12, 'Orientation','horizontal')
ylim([-1+min(E_truth(1,:)) 4+max(E_truth(1,:))])
xlabel('n', 'FontSize', 18)
ylabel('$E(X_n^1\,|\,Y_{0:n})$','FontSize', 17)
axx = gca;
axx.XAxis.FontSize = 15;
axx.YAxis.FontSize = 15;


subplot(4,1,2)
for i = 1:nsimul-1
    hp = scatter(time,E_EnKF1(:,i),5,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_EnKF1(:,nsimul),5,'filled','MarkerFaceColor','g', 'DisplayName', 'EnKF Simulations');
hold on
plot(time,E_truth(1,:),'k-','DisplayName', 'Reference','LineWidth',2)
hold on
plot(time,E_EnKF(1,:),'r-','DisplayName', 'Avg EnKF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^1$ w.r.t. the EnKF distribution');
str2 = sprintf('over %d simulations with %d ensembles ($L_2$-error $=%.2E$)',nsimul, M, L2_err_EnKF1);
title({str1,str2},'FontSize', 16)
legend show
legend('Location','northeast')
set(legend, 'FontSize', 12, 'Orientation','horizontal')
ylim([-1+min(E_truth(1,:)) 4+max(E_truth(1,:))])
xlabel('n', 'FontSize', 18)
ylabel('$E(X_n^1\,|\,Y_{0:n})$','FontSize', 17)
axx = gca;
axx.XAxis.FontSize = 15;
axx.YAxis.FontSize = 15;


subplot(4,1,3)
for i = 1:nsimul-1
    hp = scatter(time,E_ETKF1(:,i),5,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_ETKF1(:,nsimul),5,'filled','MarkerFaceColor','g', 'DisplayName', 'ETKF Simulations');
hold on
plot(time,E_truth(1,:),'k-','DisplayName', 'Reference', 'LineWidth',2)
hold on
plot(time,E_ETKF(1,:),'r-','DisplayName', 'Avg ETKF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^1$ w.r.t. the ETKF distribution');
str2 = sprintf('over %d simulations with %d ensembles ($L_2$-error $=%.2E$)',nsimul, M, L2_err_ETKF1);
title({str1,str2},'FontSize', 16)
legend show
legend('Location','northeast')
set(legend, 'FontSize', 12, 'Orientation','horizontal')
ylim([-1+min(E_truth(1,:)) 4+max(E_truth(1,:))])
xlabel('n', 'FontSize', 18)
ylabel('$E(X_n^1\,|\,Y_{0:n})$','FontSize', 17)
axx = gca;
axx.XAxis.FontSize = 15;
axx.YAxis.FontSize = 15;

subplot(4,1,4)
for i = 1:nsimul-1
    hp = scatter(time,E_Sqrt_ETKF1(:,i),5,'filled','MarkerFaceColor','g');
    hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold on
end
scatter(time,E_Sqrt_ETKF1(:,nsimul),5,'filled','MarkerFaceColor','g', 'DisplayName', 'SQRT-ETKF Simulations');
hold on
plot(time,E_truth(1,:),'k-','DisplayName', 'Reference','LineWidth',2)
hold on
plot(time,E_Sqrt_ETKF(1,:),'r-','DisplayName', 'Avg SQRT-ETKF','LineWidth',1)
hold off
str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^1$ w.r.t. the SQRT-ETKF distribution');
str2 = sprintf('over %d simulations with %d ensembles ($L_2$-error $=%.2E$)',nsimul, M, L2_err_ETKF_sqrt1);
title({str1,str2},'FontSize', 16)
legend show
legend('Location','northeast')
set(legend, 'FontSize', 12, 'Orientation','horizontal')
ylim([-1+min(E_truth(1,:)) 4+max(E_truth(1,:))])
xlabel('n', 'FontSize', 18)
ylabel('$E(X_n^1\,|\,Y_{0:n})$','FontSize', 17)
axx = gca;
axx.XAxis.FontSize = 15;
axx.YAxis.FontSize = 15;

if save_files
    export_fig(sprintf([FiguresFolderName,'/',Model,...
    '_phi=X1_dx=%d_dy=%d_N=%d_L=%d_T%d_freq_time=%d_freq_spatial=%d.pdf'],...
    dimx,dimo,N, L,T, o_freq_time, o_freq_spatial),'-m3')
end


%% plot ESS
%[mm,ind] = min(ESS_saved);
%ESS_withoutZeros = ESS_saved(1:ind-1);
figure
set(gcf, 'Position',  [100, 100, 1200, 400])
plot(1:length(ESS_saved),ESS_saved,'k-', 'MarkerSize',10,'MarkerFaceColor','k')
str = sprintf('ESS up to T = %d', T);
title(str,'FontSize',30)


if save_files
    export_fig(sprintf([FiguresFolderName,...
    '/ESS_dx=%d_dy=%d_N=%d_L=%d_T%d_freq_time=%d_freq_spatial=%d.pdf'],...
    dimx,dimo,N, L,T, o_freq_time, o_freq_spatial),'-m3')
end


%% Functions
function [E_KF, X_f, P_f]= KF(dimx,T,theta,R1,R2,C,Y, x_star, o_freq_time)
E_KF = zeros(dimx,T+1);
E_KF(:,1) = x_star;

%%% single run of KF to save X_f and P_f
X_f = zeros(dimx,T); %mean of predictor distribution, that is the mean of P(X_n | y_{0:n-1})
P_f = zeros(dimx,dimx,T); %covariance of predictor distribution...

P_a = zeros(dimx,dimx); % covariance of the filter distribution

for n = 1:T
    X_f(:,n) = theta * E_KF(:,n); %dx * 1
    P_f(:,:,n) = theta^2 * P_a +  R1;
    Vari = C*P_f(:,:,n)*C' + R2;
    %   log_Z_KF(n) =  -0.5*(dy*log2pi+logdet(Vari))-0.5*(Y(:,n) - CX_f)'/Vari * (Y(:,n) - CX_f);
    K = P_f(:,:,n) * C' /Vari;
    P_a = (eye(dimx) - K*C) * P_f(:,:,n);
    
    if mod(n, o_freq_time) == 0
        E_KF(:,n+1) = X_f(:,n) + K * (Y(:,n) -  C*X_f(:,n));
    else
        E_KF(:,n+1) = X_f(:,n);
    end
end
end


function E_loc = EnKF(M,dimx,dimo,theta,T,x_star,Y,C,sig_x,R2,R2_sqrt, o_freq_time)
    x_a = repelem(x_star,1,M); %dx x M
    E_loc = zeros(dimx,T+1);
    E_loc(:,1) = x_star;

    for n = 1:T
        dW = randn(dimx,M ); %size = dx x M
        x_f = theta * x_a + sig_x * dW; %size = dx x M
        
        if mod(n, o_freq_time) == 0
            m = sum(x_f,2)/M ; %size =dx x 1

            temp1 = (x_f - m)' * C'; % (M * dx) * (dx * dy) = M * dy
            temp2 = (x_f - m) * temp1 ; % (dx * M) * (M* dy) = dx * dy
            temp3 = C * temp2/M; % (dy * dx) * (dx * dy) = dy * dy
            temp4 = eye(dimo)/(temp3 + R2); %dy * dy
            temp5 = C' * temp4; %dx * dy
            temp6 = (x_f - m)' * temp5; % (M * dx) * (dx * dy) = M * dy
            K = (x_f - m) * temp6/M ; %(dx * M) * (M* dy) = dx * dy

            dV = randn(dimo,M); %size = dy x M
            temp7 = Y(:,n) - C * x_f - R2_sqrt * dV; %dy * M
            x_a = x_f + K * temp7; %dx * M
        else
            x_a = x_f ; %dx * M         
        end
        E_loc(:,n+1) = sum(x_a,2)/M;
    end
end


function E_loc= EtKF(M,dimx,dimo,theta,T,x_star,Y,C,sig_x, R2_sqrt_inv, o_freq_time)
    R2_sqrt_inv =  full(R2_sqrt_inv);
    x_a= repelem(x_star,1,M); %dx x M
    
    E_loc = zeros(dimx,T+1);
    E_loc(:,1) = x_star;
    
    for n = 1:T
        dW = randn(dimx,M ); %size = dx x M

        x_f = theta * x_a + sig_x *  dW; %size = dx x M
        
        if mod(n, o_freq_time) == 0
            m_f = sum(x_f,2)/M ; %size =dx x 1

            S_f = 1/sqrt(M-1) * (x_f - m_f); %dx * M
            Y_hat = C*x_f; %dy * M
            my = sum(Y_hat,2)/M; %dy * 1

            Fk = 1/sqrt(M-1) * (Y_hat - my)' * R2_sqrt_inv ; %M * dy

            Tk = Fk'*Fk + eye(dimo); %dy * dy
            K = S_f*(Fk/Tk); %dx * dy

            m_a = m_f + K * (R2_sqrt_inv *(Y(:,n) - my)); %this step was corrected
            % it is wrong in the manual "Data assimilation toolbox for Matlab"
            [Un, Dn, ~] = svd(Fk*Fk');
            S_a = S_f * Un/sqrt(Dn+eye(M)); %dx * M

            x_a = sqrt(M-1) * S_a + m_a;
        else
            x_a = x_f;
        end
        E_loc(:,n+1) = sum(x_a,2)/M;
    end
end

function E_loc= EtKF_sqrt(M,dimx,theta,T,x_star,Y,C,sig_x, R2, o_freq_time)
    R2 = full(R2);
    x_a= repelem(x_star,1,M); %dx x M
    E_loc = zeros(dimx,T+1);
    E_loc(:,1) = x_star;
    
    for n = 1:T
        dW = randn(dimx,M ); %size = dx x M
        x_f =  theta * x_a + sig_x * dW; %size = dx x M
        
        if mod(n, o_freq_time) == 0
            m_f = sum(x_f,2)/M ; %size =dx x 1
            Xfp = x_f - m_f;
            Cxf = C*x_f; %dy * M
            my = sum(Cxf,2)/M;
            S = Cxf - my; %dy * M

            invR2_S = R2\S; %dy * M inv(R2)*S
            invTTt = symmetric((M-1) * eye(M) + S' * invR2_S); %M * M..see symetric function

            [U_T, Sigma_T] = eig(invTTt);
            Tm = U_T * (sqrt(Sigma_T) \ U_T');

            Xap = sqrt(M-1) * Xfp * Tm;
            m_a = m_f + Xfp * (U_T * (Sigma_T \ (U_T' * (invR2_S' * (Y(:,n) - my)))));

            x_a = Xap + m_a;                    
        else
            x_a = x_f;
        end
        E_loc(:,n+1) = sum(x_a,2)/M;
    end
end


function B = symmetric(A)
    B = triu(A) + triu(A,1)';
end

function [path0, ESS_saved0] = LaggedPF(h,log2pi,L,dimx,dimo,theta,N,T,Y,x_star,phi1,...
    bisection_nmax,sig_x,sig_y,ldet_R1, ldet_R2,...
    ESS_min, MCMC_steps_min,MCMC_steps_max, avg_accep_rate_min,...
    avg_accep_rate_max, sig, X_f, P_f,C,o_freq_time)

    zero_vec_dx = zeros(dimx,1);
    I_dx = eye(dimx);
    phi = zeros(dimx,1); %because I am using parfor loop, I need to specify 
                    %the array size, size of phi at most dx.
    phi(1) = phi1;
    ESS_counter = 0;
    ESS_saved0 = ones((T+1)*dimx,1) * N; %its size in first dim could be larger
    path0 = zeros(dimx,N*(T+1));
    clogx = -0.5 * sig_x^(-2);
    clogy = -0.5 * sig_y^(-2);
    lw_old = -log(N)*ones(N,1);
    for n = 1:T
        if n == 1
            %at n = 1
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            i1 = 1;
            i2 = N;
            path0(:,i1:i2) =  repelem(x_star,1,N);

            %Sample the new state X_n given X_{n-1}
            %dW = randn(dimx,N); %size = dx x N
            dW = mvnrnd(zero_vec_dx,I_dx,N)';
            tem = theta * x_star; %if A is a full matrix, tem = theta * A *x_star;
            X = repelem(tem,1,N);
            X = X + sig_x * dW;
            
            if mod(n,o_freq_time )== 0
                %%%%%%%%%%%%%%%%%%%%%%%%
                % initial step in SMC sampler
                %calculate the weight w_1

                Y1 = Y(:,1);

                %since R2_sqrt = sig_y * eye(dy);

                vec = Y1 - C*X; %yn - mu, mu is mean of g(yn|xn);

                nc = -0.5*phi(1)*(dimo*log2pi+ldet_R2);
                c1 = clogy * phi(1);
                lw = nc + c1  * sum(vec.^2,1)'; %g(yn|xn)^phi(1)

                %normalize the weight
                max_lw = max(lw);
                We0 = exp(lw - max_lw);
                sumWe0 = sum(We0);
                We = We0/sumWe0;

                %resample
                ESS = sumWe0^2/sum(We0.^2);
                ESS_counter = ESS_counter+1;
                ESS_saved0(ESS_counter) = ESS;
                if ESS <= ESS_min
                    An = randsample(N,N,true,We);
                    X = X(:,An); %size = dx * N
                    lw_old = -log(N)*ones(N,1);
                else
                    lw_old = log(We);
                end

                terminate = 0;
                k = 1;
                while terminate == 0

                    %calculate the weight w_k
                    %%%%%%%%%%%%%%% Calculate phi(n+1) %%%%%%%%%%%%%%%
                     func = @(delta) fun(delta,log2pi, X, Y1, ESS_min,...
                                dimo, C, ldet_R2, clogy);
                     [delta, converged] = bisection(func,0,1,1e-5,1e-3, bisection_nmax);
                     if ~converged && k ==1
                         delta = 0.0001;
                     end
                     if (~ converged) && (k > 1)
                        delta = 1.2 * (phi(k) - phi(k - 1));
                     end
                     if delta <= 1 - phi(k) && delta >= 0
                        phi(k+1) = phi(k) + delta;
                     elseif delta > 1 - phi(k) 
                         %fprintf('delta = %.5f\n', delta);
                         delta = 1 - phi(k);
                         phi(k+1) = 1;
                         terminate = 1; %This is to stop the algorithm
                     end

                     if phi(k+1) == 1
                         terminate = 1;
                     end


                    vec = Y1 - C*X; %yn - mu, mu is mean of g(yn|xn);

                    %since R2_sqrt = sig_y * eye(dy);
                    nc = -0.5* delta *(dimo*log2pi+ldet_R2);
                    c1 = clogy * delta;
                    lw = lw_old + nc + c1 * sum(vec.^2,1)'; %g(yn|xn)^phi(1)

                    %clear TempMatrix
                    %normalize the weight
                    max_lw = max(lw);
                    We0 = exp(lw - max_lw);
                    sumWe0 = sum(We0);
                    We = We0/sumWe0;
                    %estimate the Z_k(2);
                    %log_Z_k(k+1) = log(sumWe0/N) + max_lw;

                    %resample
                    resampled = 0;
                    ESS = sumWe0^2/sum(We0.^2);
                    ESS_counter = ESS_counter+1;
                    ESS_saved0(ESS_counter) = ESS;
                    if ESS <= ESS_min
                        resampled = 1;
                        An = randsample(N,N,true,We);
                        X_hat = X(:,An); %size = dx * N
                        lw_old = -log(N)*ones(N,1);
                    else
                        lw_old = log(We);
                        X_hat = X;
                    end

                    %fprintf('sum = %.3f\n', sum(We0>0.1))
                    %%%%%%% MCMC steps - random walk %%%%%%%% 
                    accep_count = zeros(N,1);
                    avg_accep_rate = 0;
                    iter = 0;
                    terminateMCMC = 0;
                    while terminateMCMC == 0 
                        iter = iter + 1;    
                        if iter > MCMC_steps_min && avg_accep_rate >= ...
                                avg_accep_rate_min && avg_accep_rate <= avg_accep_rate_max 
                            terminateMCMC = 1;
                        end

                        if iter >= MCMC_steps_max
                            terminateMCMC = 1;
                        end

%                         covm = sig * eye(dimx);
%                         if avg_accep_rate < avg_accep_rate_min
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+10);
%                         end
%                         if avg_accep_rate > avg_accep_rate_max
%                             covm = sig*(phi(k)+2)/(phi(k)+1);
%                         end
                        sigm = sig;
                        if avg_accep_rate < avg_accep_rate_min
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+10);
                        end
                        if avg_accep_rate > avg_accep_rate_max
                            sigm = sig*(phi(k)+2)/(phi(k)+1);
                        end

    % try to put for j outside
                        %Xp = mvnrnd(X_hat',covm)'; %mvnrnd(X',sig)' 
                                    %is equiv to mvnrnd(X',sig,N)' if X a matrix
                        %Xp = X_hat + sqrt(sigm) * randn(dimx,N);
                        Xp = mvnrnd(X_hat',sigm*I_dx)';
                        vec = Y1 - C*Xp;

                        ratio =  clogy * phi(k+1) * sum(vec.^2,1);

                        vec   = Y1 - C*X_hat;
                        ratio = ratio - clogy * phi(k+1)  * sum(vec.^2,1);
                        vec = Xp - tem;
                        ratio = ratio + clogx * sum(vec.^2,1);
                        vec = X_hat - tem;
                        ratio = ratio- clogx * sum(vec.^2,1);
                        log_accep = min(0, ratio);

                        for j = 1 :N
                            if log(rand) < log_accep(j)
                                X_hat(:,j) = Xp(:,j);
                                accep_count(j) = accep_count(j) + 1;
                            end
                        end
                        accep_rate = accep_count/iter;
                        avg_accep_rate = mean(accep_rate);
                    end
                    fprintf('h = %d, MCMCsteps = %d, accep_rate_avg = %.4f\n',h, iter, avg_accep_rate)
                    X = X_hat;  
                    k = k+1;
                end

                fprintf('\n\n\n h = %d, p(1) = %d \n\n\n',h, k-1) 
                %Output: X, We

                %resample the output X using the last-calculated weights "We": If we
                %resampled using We before then We = 1/N for all particles and the next step will do
                %nothing:
                if resampled ~= 1
                    if ESS <= ESS_min
                        An = randsample(N,N,true,We);
                        X = X(:,An); %size = dx * N
                        lw_old = -log(N)*ones(N,1);
                    else
                        lw_old = log(We);
                    end
                end
            end
            i1 = N*n+1;
            i2 = N*(n+1);
            path0(:,i1:i2) = X;
          
        %% n >= 2 and <= L
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elseif n >= 2 && n <= L

            %sample from f(X_n|X_{n-1})
            %dW = randn(dimx,N); %size = dx x N

            dW = mvnrnd(zero_vec_dx,I_dx,N)';
            X = theta * X + sig_x * dW;
            
            if mod(n,o_freq_time) == 0
                %%%%%%%%%%%%%%%%%%%%%%%%
                % initial step in SMC sampler
                Yn = Y(:,n);
                %calculate the weight w_1
                vec = Yn - C*X;
                nc = -0.5*phi(1)*(dimo*log2pi+ldet_R2);
                c1 = clogy * phi(1);
                lw = lw_old + nc + c1 * sum(vec.^2,1)';

                max_lw = max(lw);
                We0 = exp(lw - max_lw);
                sumWe0 = sum(We0);
                We = We0/sumWe0;

                %estimate the Z_1(1);
                %log_Z_k(1) = log(sumWe0/N) + max_lw;

                %resample
                ESS = sumWe0^2/sum(We0.^2);
                ESS_counter = ESS_counter+1;
                ESS_saved0(ESS_counter) = ESS;
                if ESS <= ESS_min
                    An = randsample(N,N,true,We);
                    X = X(:,An); %size = dx * N
                    lw_old = -log(N)*ones(N,1);
                else
                    lw_old = log(We);
                end

                %fprintf('sum = %.3f\n', sum(We0>0.1))

                %ESS = sumWe0^2/sum(We0.^2);
                %fprintf('n = %d, k = 1, ESS = %.3f\n', n, ESS)

                terminate = 0;
                k = 1;
                while terminate == 0

                    %calculate the weight w_k
                    %%%%%%%%%%%%%%% Calculate phi(n+1) %%%%%%%%%%%%%%%
                    func = @(delta) fun(delta,log2pi, X, Yn, ESS_min, dimo,...
                                         C, ldet_R2, clogy);
                    [delta, converged] = bisection(func,0,1,1e-5,1e-3, bisection_nmax);
                    if ~converged && k ==1
                        delta = 0.0001;
                    end
                    if (~ converged) && (k > 1)
                       delta = 1.2 * (phi(k) - phi(k - 1));
                    end

                    if delta <= 1 - phi(k) && delta >= 0
                        phi(k+1) = phi(k) + delta;
                    elseif delta > 1 - phi(k) 
                        %fprintf('delta = %.5f\n', delta);
                        delta = 1 - phi(k);
                        phi(k+1) = 1;
                        terminate = 1; %This is to stop the algorithm
                    end

                    if phi(k+1) == 1
                        terminate = 1;
                    end

                    vec = Yn - C*X; %yn - mu, mu is mean of g(yn|xn);

                    %since R2_sqrt = sig_y * eye(dy);
                    nc = -0.5* delta *(dimo*log2pi+ldet_R2);
                    c1 = clogy * delta;
                    lw = lw_old + nc + c1 * sum(vec.^2,1)';
                    %clear TempMatrix

                    max_lw = max(lw);
                    We0 = exp(lw - max_lw);
                    sumWe0 = sum(We0);
                    We = We0/sumWe0;

                    %estimate the Z_n(k+1);
                    %log_Z_k(k+1) = log(sumWe0/N) + max_lw;

                    %resample
                    resampled = 0;
                    ESS = sumWe0^2/sum(We0.^2);
                    ESS_counter = ESS_counter+1;
                    ESS_saved0(ESS_counter) = ESS;
                    if ESS <= ESS_min
                        resampled = 1;
                        An = randsample(N,N,true,We);
                        X_hat = X(:,An); %size = dx * N
                        lw_old = -log(N)*ones(N,1);
                    else
                        lw_old = log(We);
                        X_hat = X;
                    end

                    %%%%%%% MCMC steps - random walk %%%%%%%% 
                    pathToUpdate = zeros(dimx,N*(n+1));
                    pathToUpdate(:,1:N*n) = path0(:,1:N*n);
                    i1 = N*n+1;
                    i2 = N*(n+1);
                    pathToUpdate(:,i1:i2) = X_hat;

                    log_f_p = zeros(n,N);
                    log_g_p = zeros(n,N);
                    log_f = zeros(n,N);
                    log_g = zeros(n,N); 

                    accep_count = zeros(N,1);
                    avg_accep_rate = 0;
                    iter = 0;
                    terminateMCMC = 0;

                    while terminateMCMC == 0 
                        iter = iter + 1;
                        if iter > MCMC_steps_min && avg_accep_rate >= avg_accep_rate_min && ...
                                        avg_accep_rate <= avg_accep_rate_max 
                            terminateMCMC = 1;
                        end

                        if iter >= MCMC_steps_max
                            terminateMCMC = 1;
                        end

%                         if avg_accep_rate < avg_accep_rate_min
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+15);
%                         elseif avg_accep_rate > avg_accep_rate_max
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/log(iter+1);
%                         else
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/2;
%                         end
%                         covm = covm * eye(dimx);
                        
%                         path_p = mvnrnd(pathToUpdate', covm)';

                        if avg_accep_rate < avg_accep_rate_min
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+15);
                        elseif avg_accep_rate > avg_accep_rate_max
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/log(iter+1);
                        else
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/2;
                        end
                        
                        %path_p = pathToUpdate + sqrt(sigm) * randn(dimx,N*(n+1));
                        path_p = mvnrnd(pathToUpdate', sigm*I_dx)';
                        
                        for v = 1:n
                            i1 = N*(v-1)+1;
                            i2 = N*v;
                            %fXp = theta * path_p(:,i1:i2); %since A = eye(dx)
                            j1 = N*v+1;
                            j2 = N*(v+1);
                            vec = path_p(:,j1:j2) - theta * path_p(:,i1:i2);
                            log_f_p(v,:) = clogx * sum(vec.^2,1);
                            
                            if mod(v,o_freq_time)==0
                                vec = Y(:,v) - C*path_p(:,j1:j2);

                                if v < n
                                    log_g_p(v,:) = clogy * sum(vec.^2,1);
                                else
                                    log_g_p(v,:) = clogy * phi(k+1) * sum(vec.^2,1);
                                end
                            end
                            %fXp = theta * pathToUpdate(:,i1:i2);
                            vec = pathToUpdate(:,j1:j2) - theta * pathToUpdate(:,i1:i2);
                            log_f(v,:) = clogx * sum(vec.^2,1);
                            
                            if mod(v,o_freq_time)==0
                                vec = Y(:,v) - C*pathToUpdate(:,j1:j2); 

                                if v < n
                                    log_g(v,:) = clogy * sum(vec.^2,1);
                                else
                                    log_g(v,:) = clogy * phi(k+1) * sum(vec.^2,1);
                                end     
                            end
                        end

                        % calculate the log acceptance prob
                        log_accep = min(0, sum(log_f_p,1) + sum(log_g_p,1)...
                                                   - sum(log_f,1) - sum(log_g,1));
                        for j = 1:N
                            if log(rand) < log_accep(j)
                                for v = 1:n
                                    i1 = N*(v-1)+j;
                                    pathToUpdate(:,i1) = path_p(:,i1);
                                end
                                accep_count(j) = accep_count(j) + 1;
                            end
                        end
                        accep_rate = accep_count/iter;
                        avg_accep_rate = mean(accep_rate);
                    end
                    if mod(k,5) == 0 %print every 5 SMC_sampler steps
                        fprintf('h = %d, n = %d, k = %d, MCMCsteps = %d, accep_rate_avg = %.4f\n',...
                                        h, n, k, iter, avg_accep_rate)
                    end

                    path0(:,1:N*(n+1)) = pathToUpdate;
                    i1 = N*n+1;
                    i2 = N*(n+1);
                    X = pathToUpdate(:,i1:i2);  
                    k = k+1;
    %                 clear pathToUpdate
    %                 clear path_p
                end

                fprintf('\n\n\n h = %d, p(%d) = %d \n\n\n',h,n,k-1)

                %resample the output X using the last-calculated weights "We": If we
                %resampled using We before then We = 1/N for all particles and the next step will do
                %nothing:
                if resampled ~=1 
                    if ESS <= ESS_min
                        An = randsample(N,N,true,We);
                        for i = 1 : n+1
                            i1 = N*(i-1)+1;
                            i2 = N*i;
                            Xtemp = path0(:,i1:i2);
                            Xtemp = Xtemp(:,An);
                            path0(:,i1:i2) = Xtemp; 
                        end
                        lw_old = -log(N)*ones(N,1);
                        i1 = N*n+1;
                        i2 = N*(n+1);
                        X = path0(:,i1:i2);
                    else
                        lw_old = log(We);
                    end
                end

                %log_Z(n) = sum(log_Z_k); %estimate of Z_n/Z_{n-1};
                %clear log_Z_k
            end
            i1 = N*n+1;
            i2 = N*(n+1);
            path0(:,i1:i2) = X;

        %% n = L+1:T
        elseif n > L

            %dW = randn(dimx,N); %size = dx x N
            dW = mvnrnd(zero_vec_dx,I_dx,N)'; %size = dx x N
            X = theta * X + sig_x * dW;

            if mod(n, o_freq_time) == 0

                Yn = Y(:,n);
                X_f_n_mL_p1 = X_f(:,n-L+1); %mean of the predictor dist. P(X_{n-L+1}|y_{0:n-L}) 
                %when n = L+1, have X_f(:,2), transiting from x1 to x2. 
                P_f_n_mL_p1_inv = eye(dimx)/full(P_f(:,:,n-L+1)); %when n = L+1, have P_fn_inv = inv(P_f(:,:,2)
                ldet_P_f_n_mL_p1 = logdet(full(P_f(:,:,n-L+1)), 'chol'); %det(P_f(:,:,n-L+1));

                X_f_n_mL = X_f(:,n-L);
                P_f_n_mL_inv = eye(dimx)/full(P_f(:,:,n-L)); %when n = L+1, have P_fn_inv = inv(P_f(:,:,2)

                %calculate the weight w_1
                i1 = N*(n-L)+1;
                i2 = N*(n-L+1);
                fXN = theta * path0(:,i1:i2);

                vec = Yn - C*X; %yn -  mean of g(yn|xn);

                nc = phi(1)/2 * (ldet_R1-dimo*log2pi -ldet_R2-ldet_P_f_n_mL_p1);
                lw = nc + phi(1) * clogy * sum(vec.^2,1);
                %in the following when n = L+1, we have mu_{X_1}(X_2) (recall:
                %path0(:,:,3) = X2)
                i1 = N*(n-L+1)+1;
                i2 = N*(n-L+2);
                vec = path0(:,i1:i2) - X_f_n_mL_p1; %when n = L+1, path0(:,:,3) --> X_2
                %this is from the exponential part of $log(\mu_{n-L}(x_{n-L+1})$

                %vec2 = path0(:,i1:i2) - fXN; %when n = L+1, have x_2 - f(
                lw = lw + phi(1) *(-0.5 * sum(vec.*(P_f_n_mL_p1_inv *vec),1) ...
                             - clogx*sum((path0(:,i1:i2) - fXN).^2,1));

                lw = lw_old + lw';
                max_lw = max(lw);
                We0 = exp(lw - max_lw);
                sumWe0 = sum(We0);
                We = We0/sumWe0;

                %estimate the Z_1(1);
                %log_Z_k(1) = log(sumWe0/N) + max_lw;

                %resample
                ESS = sumWe0^2/sum(We0.^2);
                ESS_counter = ESS_counter+1;
                ESS_saved0(ESS_counter) = ESS;
                if ESS <= ESS_min
                    An = randsample(N,N,true,We);
                    X = X(:,An); %size = dx * N
                    lw_old = -log(N)*ones(N,1);
                else
                    lw_old = log(We);
                end

                terminate = 0;
                k = 1;
                while terminate == 0
                    %calculate the weight w_1
                    %calculate the weight w_k
                    %%%%%%%%%%%%%%% Calculate phi(n+1) %%%%%%%%%%%%%%%
                    func = @(delta) fun1(delta,log2pi,theta,clogy,clogx, X,...
                        Yn, path0, X_f_n_mL_p1, P_f_n_mL_p1_inv,...
                        ldet_P_f_n_mL_p1, ESS_min, N, n,L, dimo,...
                        C, ldet_R1, ldet_R2);
                    [delta, converged] = bisection(func,0,1,1e-5,1e-3,bisection_nmax);
                    if ~converged && k ==1
                        delta = 0.0001;
                    end
                    if (~ converged) && (k > 1)
                       delta = 1.2 * (phi(k) - phi(k - 1));
                    end

                    if delta <= 1 - phi(k) && delta >= 0
                        phi(k+1) = phi(k) + delta;
                    elseif delta > 1 - phi(k) 
                        %fprintf('delta = %.5f\n', delta);
                        delta = 1 - phi(k);
                        phi(k+1) = 1;
                        terminate = 1; %This is to stop the algorithm
                    end

                    if phi(k+1) == 1
                        terminate = 1;
                    end


                    vec = Yn - C*X; %yn -  mean of g(yn|xn);

                    nc = delta/2 * (ldet_R1 - dimo*log2pi-ldet_R2...
                                    -ldet_P_f_n_mL_p1);
                    lw =  nc + delta * clogy *sum(vec.^2,1);
                    i1 = N*(n-L+1)+1;
                    i2 = N*(n-L+2);
                    vec = path0(:,i1:i2) - X_f_n_mL_p1; %when n = L+1, X_2 --> path(:,:,3)
                    %vec2 = path0(:,i1:i2) - fXN; %when n = L+1, have x_2 - theta_1 * x_1
                    % in the nc, 2pi cancels out from mu and f.
                    lw = lw + delta *  (-0.5 *sum(vec.*(P_f_n_mL_p1_inv *vec),1) ...
                                    - clogx*sum((path0(:,i1:i2) - fXN).^2,1));

                    lw = lw' + lw_old;
                    max_lw = max(lw);
                    We0 = exp(lw - max_lw);
                    sumWe0 = sum(We0);
                    We = We0/sumWe0;

                    %resample
                    resampled = 0;
                    ESS = sumWe0^2/sum(We0.^2);
                    ESS_counter = ESS_counter+1;
                    ESS_saved0(ESS_counter) = ESS;
                    if ESS <= ESS_min
                        resampled = 1;
                        An = randsample(N,N,true,We);
                        X_hat = X(:,An); %size = dx * N
                        lw_old = -log(N)*ones(N,1);
                    else
                        lw_old = log(We);
                        X_hat = X;
                    end

                    pathToUpdate = zeros(dimx,N*(L+1));

                    i1 = N*(n-L)+1;
                    i2 = N*n;
                    pathToUpdate(:,1:N*L) = path0(:,i1:i2); %x_{n-L} : x_{n-1}
                    i1 = N*L+1;
                    i2 = N*(L+1);
                    pathToUpdate(:,i1:i2) = X_hat; %x_{n}..,so pathToUpdate is X_{n-L}: X_n
                    %when n = L+1, pathToUpdate(:,:,1) = X_1 (on computer is X_2) and pathToUpdate(:,:,L+1) = X_{L+1}
                    %when n = L+2, pathToUpdate(:,:,1) = X_2 (on computer is X_3) and pathToUpdate(:,:,L+1) = X_{L+2}


                    log_f_p = zeros(L,N);
                    log_f = zeros(L,N);
                    log_g_p = zeros(L,N);
                    log_g = zeros(L,N);
                    %%%%%%% MCMC steps - random walk %%%%%%%% 
                    accep_count = zeros(N,1);
                    avg_accep_rate = 0;
                    iter = 0;
                    terminateMCMC = 0;

                    while terminateMCMC == 0 
                        %fprintf("iter = %d, acc = %.3f, min = %d\n",iter, avg_accep_rate, MCMC_steps_min)
                        iter = iter + 1;

                        if iter > MCMC_steps_min && avg_accep_rate >= avg_accep_rate_min && ...
                                    avg_accep_rate <= avg_accep_rate_max 
                            terminateMCMC = 1;
                        end

                        if iter >= MCMC_steps_max
                            terminateMCMC = 1;
                        end

%                         if avg_accep_rate < avg_accep_rate_min
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+20);
%                         elseif avg_accep_rate > avg_accep_rate_max
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/log(iter+1);
%                         else
%                             covm = sig*(phi(k)+2)/(phi(k)+1)/2;
%                         end
%                         covm = covm * eye(dimx);
% 
%                         %sample L+1 points from the proposal
%                         path_p = mvnrnd(pathToUpdate', covm)';

                        if avg_accep_rate < avg_accep_rate_min
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/(iter^2+20);
                        elseif avg_accep_rate > avg_accep_rate_max
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/log(iter+1);
                        else
                            sigm = sig*(phi(k)+2)/(phi(k)+1)/2;
                        end

                        %sample L+1 points from the proposal
                        %path_p = pathToUpdate + sqrt(sigm) * randn(dimx,N*(L+1));
                        path_p = mvnrnd(pathToUpdate', sigm*eye(dimx))';

                        if n == L+1
                            %when n = L+1, path_p(:,:,1) or pathToUpdate(:,:,1)
                            %corresponds to X_1

                            %calc log[f(x_1^{'j}|x_0)] and log[f(x_1^j|x_0)]
                            vec = path_p(:,1:N) - tem;
                            lf10p = clogx * sum(vec.^2,1);
                            vec = pathToUpdate(:,1:N) - tem;
                            lf10 = clogx * sum(vec.^2,1);

                            for v = 1:L
                                i1 = N*(v-1)+1;
                                i2 = N*v;
                                j1 = N*v+1;
                                j2 = N*(v+1);
                                vec = path_p(:,j1:j2) - theta * path_p(:,i1:i2);
                                log_f_p(v,:) = clogx * sum(vec.^2,1);
                                
                                if mod(v,o_freq_time)==0
                                    vec = Y(:,v) - C*path_p(:,i1:i2);
                                    log_g_p(v,:) = clogy * sum(vec.^2,1);
                                else
                                    log_g_p(v,:) = zeros(1,N);
                                end
                                
                                vec = pathToUpdate(:,j1:j2) - theta * pathToUpdate(:,i1:i2);
                                log_f(v,:) = clogx * sum(vec.^2,1);

                                if mod(v,o_freq_time)==0
                                    vec = Y(:,v) - C*pathToUpdate(:,i1:i2); 
                                    log_g(v,:) = clogy * sum(vec.^2,1); 
                                else
                                    log_g(v,:) = zeros(1,N);
                                end  
                            end

                            %calculate the log(the ratio to the power phi(k+1))
                            i1 = N+1;
                            i2 = N*2;
                            j1 = N*L+1;
                            j2 = N*(L+1);

                            if mod(n,o_freq_time)==0
                                vec = Y(:,n) - C*path_p(:,j1:j2);
                                log_g_pr = clogy * sum(vec.^2,1); 

                                vec = Y(:,n) - C*pathToUpdate(:,j1:j2);
                                log_gr = clogy * sum(vec.^2,1); 
                                
                            else
                                log_g_pr = zeros(1,N);
                                log_gr = zeros(1,N);
                            end
                            
                            vec = path_p(:,i1:i2) - X_f_n_mL_p1; 
                            log_mu1_p = -0.5 * sum(vec.*(P_f_n_mL_p1_inv * vec),1); 
                            vec = pathToUpdate(:,i1:i2) - X_f_n_mL_p1;
                            log_mu1 = -0.5 * sum(vec .* (P_f_n_mL_p1_inv * vec),1); 

                            log_ratio = phi(k+1) * (log_g_pr + log_mu1_p + log_f(1,:) ...
                                                    - log_gr - log_mu1 -log_f_p(1,:));

                            log_accep = min(0, log_ratio + sum(log_f_p,1) + ...
                                                lf10p + sum(log_g_p,1)...
                                                 - sum(log_f,1) - lf10 - sum(log_g,1));
                        else %n = L+2, L+3,...
                            %when n = L+2, path_p(:,:,1) or pathToUpdate(:,:,1)
                            %corresponds to X_2 and path_p(:,:,L+1) correp to
                            %X_{L+2}
                            for v = 1:L
                                i1 = N*(v-1)+1;
                                i2 = N*v;
                                j1 = N*v+1;
                                j2 = N*(v+1); 
                                vec = path_p(:,j1:j2) - theta * path_p(:,i1:i2);
                                log_f_p(v,:) = clogx * sum(vec.^2,1);

                                if mod(v+n-L-1,o_freq_time)==0
                                    vec = Y(:,v+n-L-1) - C*path_p(:,i1:i2); 
                                    %when v=L --> Y(:,n-1) - C*path_p(:,N*(L-1)+1: N*L)
                                    log_g_p(v,:) = clogy * sum(vec.^2,1);
                                else
                                    log_g_p(v,:) = zeros(1,N);
                                end
                                vec = pathToUpdate(:,j1:j2) - theta * pathToUpdate(:,i1:i2);
                                log_f(v,:) = clogx * sum(vec.^2,1);

                                if mod(v+n-L-1,o_freq_time)==0
                                    vec = Y(:,v+n-L-1) - C*pathToUpdate(:,i1:i2);
                                    log_g(v,:) = clogy * sum(vec.^2,1);
                                else
                                    log_g(v,:) = zeros(1,N);
                                end
                            end

                            %calculate the log(ratio to the power phi(k+1))
                            i1 = N+1;
                            i2 = N*2;
                            j1 = N*v+1;
                            j2 = N*(v+1); 

                            if mod(n,o_freq_time)==0
                                vec = Y(:,n) - C*path_p(:,j1:j2);
                                log_g_pr = clogy * sum(vec.^2,1);

                                vec = Y(:,n) - C*pathToUpdate(:,j1:j2);
                                log_gr = clogy * sum(vec.^2,1);
                            else
                                log_g_pr = zeros(1,N);
                                log_gr = zeros(1,N);
                            end
                            
                            vec = path_p(:,i1:i2) - X_f_n_mL_p1;
                            log_mu2_p = -0.5 * sum(vec .* (P_f_n_mL_p1_inv * vec),1);
                            vec = pathToUpdate(:,i1:i2) - X_f_n_mL_p1; 
                            log_mu2 = -0.5 * sum(vec.* (P_f_n_mL_p1_inv * vec),1);

                            log_ratio = phi(k+1) * (log_g_pr + log_mu2_p + log_f(1,:) ...
                                                    - log_gr - log_mu2 -log_f_p(1,:));
                            %calculate %\mu_{n-L-1}(x_{n-L})
                            vec = path_p(:,1:N) - X_f_n_mL;
                            log_mu1_p = -0.5 * sum(vec.* (P_f_n_mL_inv * vec),1);
                            vec = pathToUpdate(:,1:N) - X_f_n_mL;
                            log_mu1 = -0.5 * sum(vec.* (P_f_n_mL_inv * vec),1);

                            log_accep = min(0, log_ratio + log_mu1_p +...
                                            sum(log_f_p,1) + sum(log_g_p,1)...
                                          - log_mu1 - sum(log_f,1) - sum(log_g,1));
                        end

                        for j =1:N
                            if log(rand) < log_accep(j)
                                for v = 1:L+1
                                    i1 = N*(v-1)+j;
                                    pathToUpdate(:,i1) = path_p(:,i1);
                                end
                                accep_count(j) = accep_count(j) + 1;
                            end
                        end

                        accep_rate = accep_count/iter;
                        avg_accep_rate = mean(accep_rate);
    %                     clear('log_U','log_accep','log_mu1','log_mu1_p',...
    %                         'vec_mu1','vec_mu1_p','log_ratio','log_mu2',...
    %                         'log_mu2_p','log_gr','log_g_pr','vec_mu2',...
    %                         'vec_mu2_p','log_g','log_f','log_g_p','log_f_p',...
    %                         'vec_g','vec_f', 'fXp','vec_f_p','vec_g_p',...
    %                         'vec_gr','vec_g_pr')
                    end
                    if mod(k,5) == 0 %print every 5 SMC_sampler steps
                        fprintf('h = %d, n = %d, k = %d, MCMCsteps = %d, accep_rate_avg = %.4f\n',...
                            h, n, k, iter,avg_accep_rate)
                    end

                    i1 = N*(n-L) + 1;
                    i2 = N*(n+1);
                    %path0(:,:,n-L+1:n+1) = pathToUpdate;
                    path0(:,i1:i2) = pathToUpdate;
                    i1 = N*L+1;
                    i2 = N*(L+1);
                    X = pathToUpdate(:,i1:i2); 
                    k = k+1;
    %                 clear pathToUpdate 
    %                 clear path_p
                    if terminate == 0
                        i1 = N*(n-L)+1;
                        i2 = N*(n-L+1);
                        fXN = theta * path0(:,i1:i2);
                    end
                end

                fprintf('\n\n\n h = %d, p(%d) = %d \n\n\n',h ,n,k-1) 

                %resample the output X using the last-calculated weights "We": If we
                %resampled using We before then We = 1/N for all particles and the next step will do
                %nothing:
                if resampled ~= 1
                    if ESS <= ESS_min
                        An = randsample(N,N,true,We);
                        for i = 1 : n+1
                            i1 = N*(i-1)+1;
                            i2 = N*i;
                            Xtemp = path0(:,i1:i2);
                            Xtemp = Xtemp(:,An);
                            path0(:,i1:i2) = Xtemp; 
                        end
                        lw_old = -log(N)*ones(N,1);
                        i1 = N*n+1;
                        i2 = N*(n+1);
                        X = path0(:,i1:i2);
                    else
                        lw_old = log(We);
                    end
                end

                %log_Z(n) = sum(log_Z_k); %estimate of Z_n/Z_{n-1};
                %clear log_Z_k
            end
            i1 = N*n+1;
            i2 = N*(n+1);
            path0(:,i1:i2) = X;
        end 
    end
end


function y = fun(delta,log2pi, X, Yn, ESS_min, dy, C, ldet_R2, clogy)

    vec = Yn - C*X;

    %lw(i) = -0.5* delta * vec' * R2_inv * vec; %g(yn|xn)^delta
    nc = -0.5*delta*(dy*log2pi+ldet_R2);
    lw = nc + clogy * delta * sum(vec.^2,1);
                 %g(yn|xn)^delta

    max_lw = max(lw);
    W0 = exp(lw - max_lw);
    
    y = sum(W0)^2/sum(W0.^2)-ESS_min;
end

function y = fun1(delta,log2pi,theta,clogy,clogx, X, Yn, path, X_fn, P_fn_inv,...
        ldet_P_f, ESS_min, N, n,L, dy, C, ldet_R1, ldet_R2)

    vec1 = Yn - C*X; %yn -  mean of g(yn|xn);

    i1 = N*(n-L)+1;
    i2 = N*(n-L+1);
    j1 = N*(n-L+1)+1;
    j2 = N*(n-L+2);
    vec2 = path(:,j1:j2) - X_fn; %when n = L+1, X_2 --> path(:,:,3)
    vec3 = path(:,j1:j2) - theta * path(:,i1:i2);
    nc = delta/2 * (ldet_R1-dy*log2pi-ldet_R2-ldet_P_f);
    lw = nc +  delta * (clogy * sum(vec1.^2,1) ...
                 -0.5 * sum(vec2.*(P_fn_inv*vec2),1) -clogx*sum(vec3.^2,1));
    max_lw = max(lw);
    W0 = exp(lw - max_lw);
    
    y = sum(W0)^2/sum(W0.^2)-ESS_min;
end

function [c, converged] = bisection(func,a,b,diff,error,bisection_nmax)
    fa = func(a);
    fb = func(b);
    if fa * fb >= 0
        fprintf("f(a) and f(b) must have different signs\n")
        fprintf(" f(a) = %.4f, f(b) = %.4f\n",fa, fb)
        fprintf('Try different values of b\n')
        iters = 0;
        b = b/20;
        fb = func(b);
        while (fa * fb >= 0) && (iters < 100)
            iters = iters + 1;
            b = b - 0.0005;
            fb = func(b);
        end
        if (fa * fb >= 0)
            fprintf('Failed to find b so that f(a)*f(b) >= 0, f(b)= %.3f \n', fb)
            c = a;
            converged = 0;
            return
        end
    end
    iters = 0;
    converged = 0;
    while ((b - a) >= diff) && (iters <= bisection_nmax)
        iters = iters + 1;
        % Find middle point
        c = (a + b) / 2;
        %Check if middle point is root
        if abs(func(c)) <= error
            converged = 1;
            break
        end
        %Decide the side to repeat the steps
        if func(c) * func(a) < 0
            b = c;
        else
            a = c;
        end
    end
    if iters > bisection_nmax
        c = a;
        converged = 0;
    end
end


function newmap = bluewhitered(m)
%BLUEWHITERED   Blue, white, and red color map.
%   BLUEWHITERED(M) returns an M-by-3 matrix containing a blue to white
%   to red colormap, with white corresponding to the CAXIS value closest
%   to zero.  This colormap is most useful for images and surface plots
%   with positive and negative values.  BLUEWHITERED, by itself, is the
%   same length as the current colormap.
%
%   Examples:
%   ------------------------------
%   figure
%   imagesc(peaks(250));
%   colormap(bluewhitered(256)), colorbar
%
%   figure
%   imagesc(peaks(250), [0 8])
%   colormap(bluewhitered), colorbar
%
%   figure
%   imagesc(peaks(250), [-6 0])
%   colormap(bluewhitered), colorbar
%
%   figure
%   surf(peaks)
%   colormap(bluewhitered)
%   axis tight
%
%   See also HSV, HOT, COOL, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.
if nargin < 1
   m = size(get(gcf,'colormap'),1);
end
bottom = [0 0 0.5];
botmiddle = [0 0.5 1];
middle = [1 1 1];
topmiddle = [1 0 0];
top = [0.5 0 0];
% Find middle
lims = get(gca, 'CLim');
% Find ratio of negative to positive
if (lims(1) < 0) && (lims(2) > 0)
    % It has both negative and positive
    % Find ratio of negative to positive
    ratio = abs(lims(1)) / (abs(lims(1)) + lims(2));
    neglen = round(m*ratio);
    poslen = m - neglen;
    
    % Just negative
    new = [bottom; botmiddle; middle];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, neglen);
    newmap1 = zeros(neglen, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap1(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
    % Just positive
    new = [middle; topmiddle; top];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, poslen);
    newmap = zeros(poslen, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
    % And put 'em together
    newmap = [newmap1; newmap];
    
elseif lims(1) >= 0
    % Just positive
    new = [middle; topmiddle; top];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, m);
    newmap = zeros(m, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
else
    % Just negative
    new = [bottom; botmiddle; middle];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, m);
    newmap = zeros(m, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
end
% 

end


function v = logdet(A, op)
%LOGDET Computation of logarithm of determinant of a matrix
%
%   v = logdet(A);
%       computes the logarithm of determinant of A. 
%
%       Here, A should be a square matrix of double or single class.
%       If A is singular, it will returns -inf.
%
%       Theoretically, this function should be functionally 
%       equivalent to log(det(A)). However, it avoids the 
%       overflow/underflow problems that are likely to 
%       happen when applying det to large matrices.
%
%       The key idea is based on the mathematical fact that
%       the determinant of a triangular matrix equals the
%       product of its diagonal elements. Hence, the matrix's
%       log-determinant is equal to the sum of their logarithm
%       values. By keeping all computations in log-scale, the
%       problem of underflow/overflow caused by product of 
%       many numbers can be effectively circumvented.
%
%       The implementation is based on LU factorization.
%
%   v = logdet(A, 'chol');
%       If A is positive definite, you can tell the function 
%       to use Cholesky factorization to accomplish the task 
%       using this syntax, which is substantially more efficient
%       for positive definite matrix. 
%
%   Remarks
%   -------
%       logarithm of determinant of a matrix widely occurs in the 
%       context of multivariate statistics. The log-pdf, entropy, 
%       and divergence of Gaussian distribution typically comprises 
%       a term in form of log-determinant. This function might be 
%       useful there, especially in a high-dimensional space.       
%
%       Theoretially, LU, QR can both do the job. However, LU 
%       factorization is substantially faster. So, for generic
%       matrix, LU factorization is adopted. 
%
%       For positive definite matrices, such as covariance matrices,
%       Cholesky factorization is typically more efficient. And it
%       is STRONGLY RECOMMENDED that you use the chol (2nd syntax above) 
%       when you are sure that you are dealing with a positive definite
%       matrix.
%
%   Examples
%   --------
%       % compute the log-determinant of a generic matrix
%       A = rand(1000);
%       v = logdet(A);
%
%       % compute the log-determinant of a positive-definite matrix
%       A = rand(1000);
%       C = A * A';     % this makes C positive definite
%       v = logdet(C, 'chol');
%
%   Copyright 2008, Dahua Lin, MIT
%   Email: dhlin@mit.edu
%
%   This file can be freely modified or distributed for any kind of 
%   purposes.
%
%% argument checking
    assert( ismatrix(A) && size(A,1) == size(A,2), ...
        'logdet:invalidarg', ...
        'A should be a square matrix of double or single class.');
    if nargin < 2
        use_chol = 0;
    else
        assert(strcmpi(op, 'chol'), ...
            'logdet:invalidarg', ...
            'The second argument can only be a string ''chol'' if it is specified.');
        use_chol = 1;
    end
    %% computation
    if use_chol
        v = 2 * sum(log(diag(chol(A))));
    else
        [~, U, P] = lu(A);
        du = diag(U);
        c = det(P) * prod(sign(du));
        v = log(c) + sum(log(abs(du)));
    end

end