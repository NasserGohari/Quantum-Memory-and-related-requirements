clc
clear
clear all
close all

% fs = 25e9; % Sampling rate (Hz)
fs = 25e9; % Sampling rate (Hz)
t =0:1/fs:50e-06 -1/fs; % Time vector when using cos
% tau=0.2e-6; %%pulse duration when using sech
tau = abs(t(end)+1/fs - t(1)); 
% t1 = -50e-6:1/fs:0; % Time vector for p1 (from -50e-6 to 0)
% t2 = 0:1/fs:50e-6; % Time vector for p2 (from 0 to 50e-6)

%T=100e-6; %%pulse duration 
%%%t =-50e-6:1/fs:50e-6 -1/fs; % Time vector when using sech
%%tua=100e-6; %%pulse durationwhen using cos 
N=20; %%defines the shap of the pulse
delta=2e6; %%total sweep range
f0=200e6; %%center frequency





CombFrq=0.03e6; %%% is the comb frequency in which the storage time will be 1/combfrq
beta=10/tau; %%is a parameter related to the temporal width of the pulse
sigma=1e-6;
% B=10/tua;
B=0.05/tau;


% p=-delta+(2/tua)*delta.*t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% since  we are planning to use a 1us pulse duration input pulse the chirp bandwidth must be 1 MHz as well
p=0.01*(1-abs(sin((pi/tau).*(t-0.5*tau)).^N)).*sin(pi*(2*f0.*t -(1/tau)*delta.*t.^2));

% p=0.13*(1-abs(sin((pi/tau).*t).^N)).*(sin(2*pi*f0.*t -pi*(2/tau)*delta.*t.^2));
% p=0.5*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*f0.*(t-0.5*tau) -(delta/beta)*(10*sech(0.2*beta.*(t-0.5*tau))))); %%%GGGGGGGGOOOOOOOOOOD
% p=0.2*(-1+exp(t/tau)).*sin(2*pi*f0.*t);

% p196=0.35*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t +(delta/beta)*(10*sech(0.2*beta.*t)))); %%%GGGGGGGGOOOOOOOOOOD
% p203=0.35*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t -(delta/beta)*(10*sech(0.2*beta.*t)))); %%%GGGGGGGGOOOOOOOOOODw
% p196=0.35*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t -(2/tau)*delta.*t.^2)); %%%GGGGGGGGOOOOOOOOOOD
% p203=0.35*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t +(2/tau)*delta.*t.^2));%%%GGGGGGGGOOOOOOOOOODw
% p=(p196+p203);

% p196=0.35*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*200.0075e6.*t -(delta/beta)*(10*sech(0.2*beta.*(t-0.5*tau))))); %%%GGGGGGGGOOOOOOOOOOD
% p203=0.35*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*199.9925e6.*t -(delta/beta)*(10*sech(0.2*beta.*(t-0.5*tau)))));%%%GGGGGGGGOOOOOOOOOODw
% p=(p196+p203);

p196=0.5*(sech((t-0.5*tau)*beta)).*sin(pi*(2*200.0075e6.*t -(delta/beta)*log(cosh(beta.*(t-0.5*tau))))); %%%GGGGGGGGOOOOOOOOOOD
p203=0.5*(sech((t-0.5*tau)*beta)).*sin(pi*(2*199.9925e6.*t -(delta/beta)*log(cosh(beta.*(t-0.5*tau)))));%%%GGGGGGGGOOOOOOOOOODw
% p=(p196+p203);
% p=0.05*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*f0.*(t-0.5*tau)));

% p=0.1*(sech(t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p=0.2*(1-(t/(0.5*tau)).^2).*sin(pi*(2*f0.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p=sech((t-tau)*beta).*sin(pi*(2*f0.*t));
% p=0.1*((t/(tau)).^2).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2));
% p=0.01*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2));
% p=0.01*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p=0.1*(1-abs(sin((pi/tau).*t).^4)).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2));

% p=0.5*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*(f0-delta).*t +(2/tau)*delta.*t.^2)); %% Positive Chirp
% p=0.5*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*(f0+delta).*t -(2/tau)*delta.*t.^2)); %% Negative Chirp

% p=0.05*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*(f0).*t -(delta/beta)*(10*sech(0.2*beta.*(t-0.5*tau)))));
% p=0.05*(sech((t-0.5*tau)*beta)).*sin(pi*(2*(f0).*t -(delta/beta)*log(cosh(beta.*(t-0.5*tau))))); 
% % % % % 
% p=0.18*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t +(2/tau)*delta.*t.^2));
% p=0.2*(sinc(0.2*(t-0.5*tau)*beta)).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2));
% p=0.26*(1-abs(sin((pi/tau).*t).^N)).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2));


% p1=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*225e6.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p11=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*175e6.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p2=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f2.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p22=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f22.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p3=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p33=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f33.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p4=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f4.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p44=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f44.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p5=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f5.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% % p=(p1+p11+p2+p22+p3+p33+p4+p44+p5);
% p=(p1+p11);
f1=180e6;
f11=185e6;
f2=190e6;
f22=195e6;
f3=200e6;
f33=205e6;
f4=210e6;
f44=215e6;
f5=220e6;
p1=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f1.*t -(2/tau)*delta.*t.^2));
p11=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f11.*t -(2/tau)*delta.*t.^2));
p2=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f2.*t -(2/tau)*delta.*t.^2));
p22=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f22.*t -(2/tau)*delta.*t.^2));
p3=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f3.*t -(2/tau)*delta.*t.^2));
% p3=0.15*(sinc(0.2*t*beta)).*sin(pi*(2*400e6.*t +(2/tau)*0e6.*t.^2));
%p3= 0.5*(1+square(2*pi*100e6.*t, 40));%.*sin(2*pi*400e6.*t);
p33=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f33.*t -(2/tau)*delta.*t.^2));
p4=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f4.*t -(2/tau)*delta.*t.^2));
p44=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f44.*t -(2/tau)*delta.*t.^2));
p5=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*f5.*t -(2/tau)*delta.*t.^2));
% p=(p1+p11+p2+p22+p3+p33+p4+p44+p5);


p_190=0.19*(sinc(0.2*t*beta)).*sin(pi*(2*196.25e6.*t -(2/tau)*delta.*t.^2));
p_200=0.15*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t -(2/tau)*delta.*t.^2));
% p_210=0.18*(sinc(0.2*t*beta)).*sin(pi*(2*210e6.*t -(2/tau)*delta.*t.^2));
% p_190=sin(pi*(2*200e6.*t));
% p_200=sin(pi*(2*199e6.*t )+ pi);
% p_210=sin(pi*(2*210e6.*t));
% p=(p_190+p_200 );



% p_1=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*201.75e6.*t + (2/tau)*delta.*t.^2));
% p_2=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*198.25e6.*t + (2/tau)*delta.*t.^2));

p_1=0.05*sin(pi*(2*201.75e6.*t));
p_2=0.05*sin(pi*(2*198.25e6.*t ));
% p=(p_1+p_2);

% p=0.12*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t-(2/tau)*delta.*t.^2));
% p=0.1*(sech(10.*t/tau)).*sin(pi*(2*200e6.*t -(delta/beta)*log(cosh(beta.*t))));
% p=0.1*(sinc(0.2*10.*t/tau)).*sin(pi*(2*200e6.*t -(delta/beta)*log(cosh(beta.*t))));

% p=0.04*(sinc(0.2*t*beta)).*sin(pi*(2*200e6.*t));
% p=0.01*sin(pi*(2*003e6.*t));
% p=(p3);
%p=0.11*(1-(t/(0.5*tau)).^2).*sin(pi*(2*f3.*t -(2/tau)*delta.*t.^2));
% p=0.11*(sinc(0.2*t*beta)).*sin(pi*(2*f3.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p=0.08*(1-(t/(0.5*tau)).^2).*sin(pi*(2*f3.*t -(delta/beta)*(10*sech(0.2*beta.*t))));

% p=0.085*(1-abs(sin((pi/tau).*t).^N)).*sin(pi*(2*f3.*t -(2/tau)*delta.*t.^2));

% p203_5=0.5*(1-(t/(0.5*tau)).^2).*sin(pi*(2*203.5e6.*t -(5e6/beta)*(10*sech(0.2*beta.*t))));
% p196_5=0.5*(1-(t/(0.5*tau)).^2).*sin(pi*(2*196.5e6.*t -(5e6/beta)*(10*sech(0.2*beta.*t))));

% p203_5=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*203.5e6.*t -(5e6/beta)*(10*sech(0.2*beta.*t))));
% p196_5=0.1*(sinc(0.2*t*beta)).*sin(pi*(2*196.5e6.*t -(5e6/beta)*(10*sech(0.2*beta.*t))));
% p=(p203_5+p196_5);


% % f0=210.5e6; %%center frequency
% % p=0.14*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(2/tau)*delta.*t.^2)); %Pos
% % p=0.11*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t +(2/tau)*delta.*t.^2)); %Neg
% 
% % p1=0.08*(sech(beta.*t)).*sin(pi*(2*f1.*t -(delta/beta)*log(cosh(beta.*t))));
% % p11=0.08*(sech(beta.*t)).*sin(pi*(2*f11.*t -(delta/beta)*log(cosh(beta.*t))));
% % p2=0.08*(sech(beta.*t)).*sin(pi*(2*f2.*t -(delta/beta)*log(cosh(beta.*t))));
% % p22=0.08*(sech(beta.*t)).*sin(pi*(2*f22.*t -(delta/beta)*log(cosh(beta.*t))));
% % p3=0.08*(sech(beta.*t)).*sin(pi*(2*f3.*t -(delta/beta)*log(cosh(beta.*t))));
% % p33=0.08*(sech(beta.*t)).*sin(pi*(2*f33.*t -(delta/beta)*log(cosh(beta.*t))));
% % p4=0.08*(sech(beta.*t)).*sin(pi*(2*f4.*t -(delta/beta)*log(cosh(beta.*t))));
% % p44=0.08*(sech(beta.*t)).*sin(pi*(2*f44.*t -(delta/beta)*log(cosh(beta.*t))));
% % p5=0.08*(sech(beta.*t)).*sin(pi*(2*f5.*t -(delta/beta)*log(cosh(beta.*t))));



% p=0.05*sin(2*pi*f0.*t);

%  p=(1-abs(sin((pi/tau).*t).^N)).*(sin(2*pi*f0.*t +(2*pi/tau)*delta.*t.^2));%%positive slope of sweep 
% 
% p=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(10*t.^2/tau^2)));%% linear chirp
% p=0.08*(sech(t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(log(cosh(beta.*t)))));%%
% p=0.08*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(log(cosh(beta.*t)))));







%  p=0.08*(1-abs(sin((pi/tua).*t)).^N).*sin(2*pi*f0.*t+2*pi*delta.*t-(2*pi/tua)*delta.*t.^2);%%negative slope of sweep 
%  p=0.0675*(sinc(0.2*t*beta)).*sin(pi*(2*f0.*t -(delta/beta)*(10*sech(0.2*beta.*t))));
% p=0.145*(1 - 2 * abs(t- (tua)/2) / (tua)).*(sin(2*pi*f0.*t-2*pi*delta.*t+(2*pi/tua)*delta.*t.^2));%% Bartlett (Triangular) Window: Almost FFT matches to the input pulse
% p=(exp(-(0.5*B.*t).^2)).*sin(2*pi*f0.*t);%%Guassian pulse for input, the input is 3us duration where has 1us pulse width 

% p=0.25*sin(2*pi*f0.*t);


%%%p=0.08*(1-abs(cos((pi/tua).*t).^N)).*sin(2*pi*fsw.*t);%%positive slope of sweep 
%%%p=0.5*(square(pi*20e6.*t,80)+1).*sin(2*pi*f0.*t);
%%%p=(1-abs(cos((pi/tua).*t).^N)).*sin(2*pi*f0.*t);

%% Remind that when using RAMP envelope the chirp range has to be equal to the pulse bandwidth 
%%%p=0.05*(sawtooth(2*pi*t/tua, 0.5)+1).*(sin(2*pi*f0.*t-2*pi*delta.*t+(2*pi/tua)*delta.*t.^2)); %%Usinfg the ramp pulse envelope which its FFT matched the Input pulse




%%p=0.08*(1-abs(cos((pi/tua).*t).^N)).*sin(2*pi*fsw.*t);
%%p=((sech(beta.*t)).^2).*sin(2*pi*f0.*t+0.1*pi*delta*tua.*tanh(beta.*t));
%%%p=((sech(beta.*t)));%%.*sin(2*pi*f0L.*t);
%%p=(exp(-0.5*(t.^2)/(sigma^2)));%%.*sin(2*pi*f0.*t);%%Guassian pulse
%p=(1-abs(cos((pi/tua).*t).^N));%%.*sin(2*pi*f0.*t-2*pi*delta.*t+(2*pi/tua)*delta.*t.^2);
%%p=1e-7*(0.5*sigma./(pi*((t).^2 +(0.5*sigma)^2)));
%%%p=0.08*(1-abs(cos((pi/tua).*t).^N)).*sawtooth(2*pi*f0.*t-2*pi*delta.*t+(2*pi/tua)*delta.*t.^2,1);
%%%p=0.08*exp((-t.^8)/69e-37);%%.*sin(2*pi*fsw.*t);
%%%p=0.2*sin(2*pi*f0L.*t);

%%p=0.25*(square(2*pi*CombFrq.*t,50)+1).*(1-abs(cos((pi/tua).*t).^N)).*sin(2*pi*f0.*t+pi*delta.*t-(pi/tua)*delta.*t.^2);
%%p=(1-abs(cos((pi/tua).*t).^N)); +pi*delta.*t-(pi/tua)*delta.*t.^2
%TBP_Gussian=2*log(2)/pi;
%Delta_nu=TBP_Gussian/tua;
%%width = fwhm(t,p)




P =fftshift(abs(fft(p))); % Fourier transform of the pulse
f = (-length(t)/2:length(t)/2-1)*fs/length(t); % Frequency axis
figure;
subplot(2,1,1);
plot(t, p); % Plot the real part of the pulse
xlabel('Time (s)');
ylabel('Amplitude');
title('Adiabatic Rapid Passage Pulse (RAP)');
subplot(2,1,2);
plot(f, P); % Plot the Fourier transform
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Fourier Transform of RAP Pulse');
% 
% % Create a table containing the plot data (excluding the 't' and 'p' headers)
% T = table(t', p');
% 
% % Save the table in a CSV file (without writing the 't' and 'p' headers)
% writetable(T, 'plotData.txt', 'WriteVariableNames', false);




% Calculate the integral of the Fourier transform over the range of the pulse
% Using the trapezoidal rule for numerical integration
pulse_integral = 0.5*trapz(f, P);

fprintf('Integral of the Fourier transform over the pulse range: %f\n', pulse_integral);






% 
% % t1= t(t<0);
% % t2=t(t>0 & t<50e-6);
% 
% % p1=zeros(size(t1));
% p1=0.11*(sinc(0.2*t1*beta)).*sin(pi*(2*f0.*t1 +2*(delta).*t1+(4/tau)*delta.*t1.^2));
% % p2=zeros(size(t2));
% p2=0.11*(sinc(0.2*t2*beta)).*sin(pi*(2*f0.*t2 +2*(delta).*t2-(4/tau)*delta.*t2.^2));
% 
% p=[p1 p2];
% t = [t1, t2];
% P = fftshift(abs(fft(p))); % Fourier transform of the pulse
% f = (-length(t)/2:length(t)/2-1) * fs / length(t); % Frequency axis
% 
% figure;
% subplot(2,1,1);
% plot(t, p); % Plot the real part of the pulse
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Adiabatic Fast Passage Pulse');
% 
% subplot(2,1,2);
% plot(f, P); % Plot the Fourier transform
% xlabel('Frequency (Hz)');
% ylabel('Amplitude');
% title('Fourier transform of Adiabatic Fast Passage Pulse');
% 
