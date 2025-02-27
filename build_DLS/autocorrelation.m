% Calculates Autocorrelation function from given data. 
% Data assumed to be in pairs (time, value).

%% LOAD DATA

pathdir= 'C:\Users\Part 2 Users\Documents\DLS\E2b\DLS_files';  %set the directory data name
cd(pathdir);

% Import data
filename= 'data2.prn';    %% name of your file
A = importdata(filename,'\t',3);
data = A.data(:,2);     %   voltage
t = A.data(:,1);        %   time steps

% Get no. of samples.  N was previously limited to 10^6 
N=length(data);

%% calculate autocorrelation function

ac= zeros(1,N-1);     %% prepare the variable
m=mean(data);       %% mean of the data
%%%% effective calculation of the autocorrelation
for j= 1:N-1
    ac(j)= mean((data(1:end-j)-m).*(data(j+1:end)-m));
end

%%%%% plot the autocorrelation function in linear and semilogy scale
%figure(1)
%plot(t(1:end-1),ac,'.')
%title('Autocorrelation function')
%xlabel('$t  [ms]$','Interpreter','latex','FontSize',16)
%ylabel('$C(t)  [V^{2}]$','Interpreter','latex','FontSize',16 )

%figure(2)
%semilogy(t(1:end-1),ac,'.')
%title('Autocorrelation function LOGY')
%xlabel('$t  [ms]$','Interpreter','latex','FontSize',16)
%ylabel('$C(t)  [V^{2}]$','Interpreter','latex','FontSize',16 )




%% fit with exponential decay
%%%%% instead of doing an exponential fit, I do a linear fit on (t,log(ac)). 
%%%%% If data are exponential decay, this should give a line in semilogy,
%%%%% tau is the characteristic  time of the exponential 

rangemax=13;% here we set the data that you want to fit. Depends on the angle
[p,s]= polyfit((1:rangemax), log(ac(1:rangemax)),1);
tau= -1/p(1)   %%% characteristic time of exponential decay

%%%% calculate error on p(1)
ste = sqrt(diag(inv(s.R)*inv(s.R')).*s.normr.^2./s.df)



%%%% plot the autocorrelation function with the fit

figure(3);
semilogy(1:rangemax, ac(1:rangemax),'ro'); hold on;
semilogy(1:rangemax, exp(polyval(p,1:rangemax)),'b-','LineWidth',2);

xlabel('$t  [ms]$','Interpreter','latex','FontSize',16)
ylabel('$C(t)  [V^{2}]$','Interpreter','latex','FontSize',16 )
h = legend('Exp data', 'fit $ -e^{t/\tau}$');
set(h,'Interpreter','latex','FontSize',14);
title('Correlation function semilogy scale ','Interpreter','latex','FontSize',14);



figure(4);
rangemax= rangemax + 100;
plot(1:rangemax, ac(1:rangemax),'ro'); hold on;
plot(1:rangemax, exp(polyval(p,1:rangemax)),'b-','LineWidth',2);
xlabel('$t  [ms]$','Interpreter','latex','FontSize',16);
ylabel('$C(t)  [V^{2}]$','Interpreter','latex','FontSize',16 );
h = legend('Exp data', 'fit $ -e^{t/\tau}$');
set(h,'Interpreter','latex','FontSize',14);
title('Correlation function linear scale','Interpreter','latex','FontSize',14);






