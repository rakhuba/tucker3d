%-- 29.10.12, 21:08 --%
Ustoich_1
%-- 31.10.12, 13:01 --%
%-- 31.10.12, 14:30 --%
%-- 31.10.12, 14:36 --%
Ustoich_1
B
Ustoich_1
B
Ustoich_1
Ustoich_2
plot(u_0)
Ustoich_2
plot(u_0)
Ustoich_2
plot(u_0/(exp(1)*pi^2))
plot(u*(exp(1)*pi^2))
Ustoich_2
plot(u*(exp(1)))
plot(u_0)
plot(u*(exp(pi^2)))
Ustoich_2
plot(u*(exp(pi^2)))
Ustoich_2
plot(u*(exp(pi^2)))
Ustoich_2
%-- 31.10.12, 19:29 --%
Ustoich_2
bM
Ustoich_2
plot(u./u_0)
(u./u_0)
exp(-pi^2)
exp(-pi^2/2)
exp(-2*pi^2)
exp(-pi^2/2)
Ustoich_2
plot(u./u_0)
exp(-pi^2)
Ustoich_2
(u./u_0)
Ustoich_2
(u./u_0)
exp(-pi^2)
A
Ustoich_2
exp(-pi^2)
Ustoich_2
exp(-pi^2)
(u./u_0)
Ustoich_2
(u./u_0)
exp(-pi^2)
Ustoich_2
(u./u_0)
Ustoich_2
plot(l(:,5))
Ustoich_2
c
Ustoich_2
B
Ustoich_2
for k = 1:16
plot(fft(eye(k+16)))
axis equal
M(k) = getframe;
end
movie(M,30)
Ustoich_2
i_0
M
W
w
Xi(:,1)
Ustoich_2
M
sum((w+u_0).*Xi(:,3))
sum((w+u_0).*Xi(:,1))
sum((w+u_0).*Xi(:,2))
sum((w+u_0).*Xi(:,4))
sum((w+u_0).*Xi(:,5))
sum((w+u_0).*Xi(:,6))
sum((w+u_0).*Xi(:,7))
sum((w+u_0).*Xi(:,8))
sum((w+u_0).*Xi(:,9))
sum((w+u_0).*Xi(:,10))
sum((w+u_0).*Xi(:,11))
Ustoich_2
plot(u_0+w)
Ustoich_2
plot(w)
Ustoich_2
plot(u_0)
Ustoich_2
%-- 02.11.12, 0:10 --%
%-- 03.11.12, 15:45 --%
A=ones(3)
dim(A)
size(A)
size(A)(1)
M=size(A)
M(1)
Pryam_test
shallow
Pryam_test
shallow
Pryam_test
surf(x,y,eta)
Pryam_test
surf(x,y,eta)
surf(x,y,eta_0)
Pryam_test
movie(M)
Pryam_test
movie(M)
Pryam_test
movie(M)
Pryam_test
%-- 04.11.12, 12:43 --%
Pryam_test
surf(x,y,eta)
surf(x,y,eta_0)
Pryam_test
surf(x,y,eta_0)
Pryam_test
surf(x,y,eta_0)
surf(x,y,eta)
Pryam_test
surf(x,y,eta)
Pryam_test
%-- 04.11.12, 14:38 --%
Pryam_test
%-- 04.11.12, 17:08 --%
obrat_test
surf(x,y,eta)
surf(x,y,eta_obs)
surf(x,y,eta)
surf(x,y,eta_obs)
surf(x,y,eta)
obrat_test
surf(x,y,eta)
%-- 04.11.12, 17:39 --%
obrat_test
Pryam_test
surf(x,y,eta)
plot( t1_x, err_L2)
Pryam_test
obrat_test
Pryam_test
obrat_test
Pryam_test
surf(x,y,eta)
plot( t1_x, err_L2)
%-- 05.11.12, 13:51 --%
M=dlmread(test.txt)
M=dlmread('test.txt')
A=fread('test.txt')
fopen('test.txt')
A=fread(3)
A=fread(1)
A=fread(2)
A=fread(3)
clear
M=dlmread('test.txt','')
M=dlmread('test.txt','\t')
textscan(3,'integer')
A=textscan(3,'integer')
M=dlmread('test1.txt')
M=dlmread('Black Sea.txt');
M=dlmread('test.txt')
clear
M=dlmread('test.txt')
R=dlmread('Black Sea.txt');
R=dlmread('Black Sea.txt')
dlmread('Black Sea.txt')
R=dlmread('Black Sea.txt');
Pryam_BS
Size
Pryam_BS
surf(eta_0)
Pryam_BS
surf(eta_0)
Pryam_BS
surf(eta_0)
Pryam_BS
Video
Ustoich_2
%-- 06.11.12, 18:16 --%
Video
Pryam_BS
case N
switch N
case 100
x=5
case 1
x=1
end
x
switch N
case N=1
switch N
case 100
asd=1
case 1
asd=2
end
asd
switch N
case 50
asd=1
case 1
asd=5
end
kraev_mask
shallow_mask
Pryam_BS
Video
Pryam_BS
%-- 07.11.12, 1:41 --%
Pryam_BS
%-- 07.11.12, 13:21 --%
Obrat_BS
surf(eta_obs)
surf(eta_0)
Obrat_BS
surf(eta_00)
surf(eta_0)
surf(eta_0-eta_00)
surf(eta)
shading interp
surf(eta)
shading interp
surf(eta_0)
surf(eta)
for i=1:Size(1)+1
for j=1:Size(2)+1
if M(i,j)=1
eta(i,j)=-0.05;
end
end
end
for i=1:Size(1)+1
for j=1:Size(2)+1
if Mask(i,j)=1
eta(i,j)=-0.05;
end
end
end
for i=1:Size(1)+1
for j=1:Size(2)+1
if Mask(i,j)==1
eta(i,j)=-0.05;
end
end
end
for i=1:Size(1)
for j=1:Size(2)
if Mask(i,j)==1
eta(i,j)=-0.05;
end
end
end
surf(eta)
shading interp
for i=1:Size(1)
for j=1:Size(2)
if Mask(i,j)==1
eta(i,j)=-0.5;
end
end
end
surf(eta)
shading interp
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta_00)
shading interp
surf(eta_obs)
shading interp
surf(eta_obs)
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta_00-eta_0)
surf(eta_0)
surf(eta_00-eta_0)
Obrat_BS
surf(eta_00)
Obrat_BS
surf(eta)
Obrat_BS
surf(eta)
shading interp
surf(eta)
%-- 10.11.12, 10:43 --%
Obrat_BS
surf(eta)
Pryam_BS
%-- 10.11.12, 11:32 --%
Pryam_BS
Obrat_BS
%-- 12.11.12, 22:14 --%
Pryam_BS
Obrat_BS
%-- 13.11.12, 17:05 --%
Pryam_test
Pryam_BS
%-- 14.11.12, 1:43 --%
load cape
image(X,'CDataMapping','scaled')
colormap(map)
Pryam_BS
surf(eta)
interp shading
shading interp
surf(eta)
shading interp
Pryam_BS
M=Mask>0;
eta(M)=0;
surf(eta,'DisplayName','eta');figure(gcf)
shading interp
Z=eta;
Z(~M)=NaN;
surf(eta);
shading interp;
S=surf(Z);
set(S'FaceColor',[0 0 0]);
Z=eta;
Z(~M)=NaN;
surf(eta);
shading interp;
S=surf(Z);
set(S,FaceColor',[0 0 0]);
Z=eta;
Z(~M)=NaN;
surf(eta);
shading interp;
S=surf(Z);
set(S,FaceColor,[0 0 0]);
Z=eta;
Z(~M)=NaN;
surf(eta);
shading interp;
S=surf(Z);
set(S,'FaceColor',[0 0 0]);
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[0 0 0]);
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[1 1 1]);
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[0 0 0]);
M=Mask>0;
eta(M)=NaN;
surf(eta);
M=Mask>0;
eta(M)=NaN;
surf(eta);
shading interp;
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[1 1 1]);
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[1 1 1]);
axis([0 163 0 290 0 10])
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[1 1 1]);
axis([0 163 0 290 0 10])
M=Mask>0;
eta(M)=0;
Z=eta;
Z(~M)=NaN;
surf(eta);
hold on;
shading interp;
S=surf(Z);
set(S,'FaceColor',[1 1 1]);
axis([0 290 0 163 0 10])
axis([0 290 0 163 0 5])
axis([0 290 0 163 0 2])
axis([0 290 0 163 0 3])
%-- 14.11.12, 23:11 --%
Pryam_BS
M=Mask>0;
eta(M)=NaN;
surf(eta_surf);
shading interp
Pryam_BS
surf(eta_surf);
shading interp
Pryam_BS
M=Mask>0;
eta(M)=NaN;
surf(eta_surf);
shading interp
Pryam_BS
M=Mask>0;
eta(M)=NaN;
surf(eta_surf);
shading interp
M=Mask>0;
eta(M)=NaN;
surf(eta_surf);
shading interp
Pryam_BS
M=Mask_surf>0;
eta(M)=NaN;
surf(eta_surf);
shading interp
surf(Mask)
surf(Mask_surf)
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
axis off
colormap(winter)
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar('YTickLabel',...
{'metr'})
axis off
colormap(winter)
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
axis off
colormap(winter)
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
axis off
colormap(winter)
Pryam_BS
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta0);
shading interp
colorbar
legend
axis off
for i=1:Size(1)
eta_surf(i,:)=eta_0(Size(1)-i+1,:) ;
end
%surf(eta);
M=Mask_surf>0;
eta_surf(M)=NaN;
surf(eta_surf);
shading interp
colorbar
legend
axis off
Obrat_BS
surf(eta_00)
plot(L_2)
xlabel('N');
ylabel('\alpha');
ylabel('\frac{\|\Phi\|}{2}');
ylabel('\|\Phi\|');
ylabel('||\Phi ||');
ylabel('||\eta_0 -eta_0^k||');
ylabel('||\eta_0 -\eta_0^k||');
ylabel('||\eta_0 -\eta_0^k||/||\eta_0||');
ylabel('||\eta_0 -\eta_0^k|| / ||\eta_0||');
Obrat_BS
%-- 15.11.12, 22:32 --%
H = dlmread('Depth');
H = dlmread('Depth.txt');
surf(H)
size(h)
A=size(H);
Botimetry
Pryam_BS
Botimetry
Pryam_BS
shallow_mask_video
Pryam_BS
Obrat_BS
surf(eta_obs)
Obrat_BS
surf(Mask)
surf(eta_obs)
Obrat_BS
-H
max(Mes)
max(max(Mes))
Obrat_BS
surf(u)
Obrat_BS
asd
Obrat_BS
surf((Mask-1))
Obrat_BS
surf(eta)
Pryam_BS
M=Mask>0;
eta_surf(M)=NaN;
surf(eta);
shading interp
colorbar
legend
axis off
M=Mask>0;
eta(M)=NaN;
surf(eta);
shading interp
colorbar
legend
axis off
%-- 18.11.12, 0:43 --%
%-- 18.11.12, 14:39 --%
Pryam_BS
M=Mask>0;
eta_0(M)=NaN;
surf(eta_0);
shading interp
colorbar
legend
axis off
%-- 20.11.12, 23:44 --%
Pryam_test
plot(err_L2)
Pryam_test
Obrat_BS
h = figure; hold('on');
plot(xi,y,'r-', 'Linewidth', 2);
plot(xi,y,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('Time, $\xi$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Value, $y$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
h = figure; hold('on');
plot(L_2)
axis('tight');
xlabel('Time, $\xi$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Value, $y$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
Obrat_BS
axis('tight');
xlabel('ÎœÃ-◊œ …‘≈“¡√… , $N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$L_2$ Œœ“Õ¡ œ€…¬À… $\| \eta_0-\eta_0^k \|/\| \eta_0 \|$, $\eta$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('ÎœÃ-◊œ …‘≈“¡√… , $N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$L_2$ Œœ“Õ¡ œ€…¬À… $\| \eta_0-\eta_0^k \|/\| \eta_0 \|$, $\eta$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2)
plot(L_2,'b.', 'MarkerSize', 12);
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
xlabel('ÎœÃ-◊œ …‘≈“¡√… , $N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$L_2$ Œœ“Õ¡ œ€…¬À… $\| \eta_0-\eta_0^k \|/\| \eta_0 \|$, $\eta$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
xlabel(' $N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$L_2$ Œœ“Õ¡ œ€…¬À… $\| \eta_0-\eta_0^k \|/\| \eta_0 \|$, $\eta$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\| \eta_0-\eta_0^k \|/\| \eta_0 \|$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N\frac{1}{2}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\| \eta_0-\eta_0^k \|/\| \eta_0 \|$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N\frac{1}{2}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\| \eta_0-\eta_0^k \|/\| \eta_0 \|$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N\frac{1}{2}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\| \eta_0-\eta_0^k \|/\| \eta_0 \|$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
hold on
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
hold('on');
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
Obrat_BS
hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
h=figure;hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(L_2)
for i=1:3
L_2(i)=1+i
end
h=figure;hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
Obrat_BS
h=figure;hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
h=figure;hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
Tsunami
h=figure;hold('on');
plot(L_2,'r-', 'Linewidth', 2);
plot(L_2,'b.', 'MarkerSize', 12);
axis('tight');
xlabel('$N$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{\| \eta_0-\eta_0^k \|}{\| \eta_0 \|}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
plot(Tijk,normijk,'--rs');
h=figure;hold('on');
plot(Tijk,normijk,'--rs');
axis('tight');
xlabel('$t_{obs}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{J}{J_0}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
h=figure;hold('on');
plot(Tijk,normijk);
axis('tight');
xlabel('$t_{obs}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{J}{J_0}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
Tsunami
h=figure;hold('on');
plot(Tijk,normijk);
axis('tight');
xlabel('$t_{obs}$', 'FontSize', 24, 'FontName', 'Times', 'Interpreter','latex');
ylabel('$\frac{J}{J_0}$',...
'FontSize', 24, 'FontName', 'Times',...
'Interpreter','latex');
set(gca, 'FontSize', 24, 'FontName', 'Times')
saveas(h,'ModelOne.eps', 'psc2');
saveas(h,'ModelOne.png', 'png');
%-- 21.11.12, 22:17 --%
Obrat_BS
surf(eta_obs)'
Obrat_BS
surf(eta_obs)
shading interp
Obrat_BS
%-- 22.11.12, 23:11 --%
euler
%-- 23.11.12, 13:29 --%
%-- 23.11.12, 19:34 --%
%-- 23.11.12, 19:35 --%
ones(1)
ones(2)
eye(2)
obrat
eye(2)
obrat
plot(u);
obrat
%-- 23.11.12, 22:52 --%
%-- 24.11.12, 14:53 --%
obrat
plot(phi_obs)
plot(u)
obrat
%-- 26.11.12, 11:15 --%
ustoich_3
uk = u;
uk = u_0;
A = eye(N-1,N-1);
uj = A*uk;
uj = A*uk';
ujj = uk'*A;
ujj = uk*A;
ustoich_3
for i=1:i_0
c(i) = 0;
end
i_0=5
for i=1:i_0
c(i) = 0;
end
i_0 =1000;
for i=1:i_0
c(i) = 0;
end
ustoich_3
test
ustoich_3
test
ustoich_3
parab_nonlin_video
ustoich_3
test
ustoich_3
%-- 03.12.12, 15:05 --%
vrem
%-- 03.12.12, 23:08 --%
vrem
%-- 03.12.12, 23:16 --%
vrem
%-- 03.12.12, 23:34 --%
vrem
160*4 + 850*2 + 750+ 85*4
%-- 03.12.12, 23:51 --%
vrem
dot(asd,das)
x=1/0
dot(x,y)
%-- 04.12.12, 0:08 --%
clear;
scalar=0;
N = 10000000;
x = zeros(1,N);
y = zeros(1,N);
for i=1:N
ö öx(i)=1;
ö öy(i)=1;
end
tic
for i=1:N
ö öscalar = scalar + x(i)*y(i);
end
toc
tic
scal = x*y';
toc
clear;
scalar=0;
N = 10000000;
x = zeros(1,N);
y = zeros(1,N);
for i=1:N
öx(i)=1;
ö öy(i)=1;
end
tic
for i=1:N
ö öscalar = scalar + x(i)*y(i);
end
toc
tic
scal = x*y';
toc
untitled
%-- 04.12.12, 0:10 --%
vrem
untitled
%-- 04.12.12, 12:31 --%
ustoich_3
%-- 04.12.12, 15:38 --%
ustoich_3
%-- 04.12.12, 16:39 --%
ustoich_3
%-- 04.12.12, 16:54 --%
%-- 06.12.12, 22:20 --%
for i=1:1000
plot(i);
M(i)= getframe;
end
for i=1:1000
plot(i);
M(i)= getframe;
end
for i=1:1000
x(i)=i
plot(x)
M(i)= getframe;
end
y=x;
for i=1: max(x)
ö for j=1:i
ö ö öx ( j ) = ö( j );
ö ö öyy( j ) = y( j );
öend
axis( [min(x), max(x), min( y ), max ( y )])
plot(xx,yy);
MM(i) = getframe;
end
for i=1:max(x)
ö for j=1:i
ö ö öxx(j) = öx(j);
ö ö öyy(j) = y(j);
öend
axis( [min(x), max(x), min( y ), max ( y )])
plot(xx,yy);
MM(i) = getframe;
end
for i=1: max(x)
ö for j=1:i
ö ö öx ( j ) = ö( j );
ö ö öyy( j ) = y( j );
öend
axis( [min(x), max(x), min( y ), max ( y )])
plot(xx,yy);
MM(i) = getframe;
end
blabla
max(x)
min(x)
blabla
kepler
blabla
kepler
kepler1
%-- 06.12.12, 23:52 --%
'a'
' 5+5=' 5+5
'5+5='
'5+5=' 5
'5+5='
5
kepler1
'k=
text(asd)
text('asd')
kepler
%-- 13.12.12, 1:05 --%
%-- 14.12.12, 0:38 --%
sunwind
%-- 22.01.13, 19:22 --%
A = [1 2; 3 4]; permute(A,[2 1])
A = [1 2; 3 4]
A = [1 2; 3 4]; permute(A,[1 1])
A = [1 2; 3 4]; permute(A,[3 1])
A = [1 2; 3 4]; permute(A,[1 2])
A = [1 2 3; 4 5 6; 7 8 9]; permute(A,[2 1])
%-- 26.01.13, 13:37 --%
%-- 03.02.13, 21:11 --%
Video
Pryam_BS
%-- 20.02.13, 20:50 --%
A=rand([100,100]);
[u,s,v] = svd(A);
A=rand([100,100]);
[u,s,v] = svd(A);
svd_test
%-- 23.02.13, 17:40 --%
Video
Pryam_BS
%-- 03.03.13, 21:19 --%
svertka_0.0
svertka_0
pdetool
svertka_0
surf(reshape(sol,N,N))
surf(reshape(sol,N,N)-solf)
svertka_0
surf(reshape(sol,N,N)-solf)
surf(solf)
svertka_0
surf(reshape(sol,N,N))
surf(x,y,f-reshape(sol,N,N))
surf(x,y,reshape(sol,N,N)-solf)
surf(reshape(sol,N,N)-solf)
svertka_0
surf(reshape(sol,N,N)-solf)
svd(q)
[u,d,v] = svd(q)
svertka_0
[u,d,v] = svd(q);
svertka_0
[u,d,v] = svd(q);
svertka_0
surf(reshape(sol,N,N)-solf)
svertka_0
int(sin(2*x), 0, pi/2)
function y = myfun(x)
y = 1./(x.^3-2*x-5);
poiss_int
svertka_0
surf(solf)
svertka_0
%-- 12.03.13, 1:12 --%
svertka_0
surf(solf)
svertka_0
surf(solf)\
%-- 17.03.13, 13:39 --%
%-- 12.04.13, 17:23 --%
A=rand([100,100]);
u,s,v = svd(A)
[u,s,v] = svd(A)
print(s)
print(diag(s))
for i=1:100
S(i) = s(i,i)
end
print(S)
plot(s)
plot(S)
B = A.^3
[u1,s1,v1] = svd(B);
for i=1:100
S1(i) = s1(i,i);
end
plot(S1)
plot(S,S1)
1:100
plot(1:100,S,1:100,S1)
plot(S1-S)
C = A^3;
[u2,s2,v2] = svd(C);
for i=1:100
S2(i) = s2(i,i);
end
plot(S2)
plot(S1-S)
C = A.^(1/3)
experiment
clear
u = rand(10)
u = rand([10,1])
v = rand([10,1])
A = u*v'
[l,s,m]=svd(A);
experiment
svertka_0
experiment
%-- 15.04.13, 14:33 --%
svertka_0
surf(q)
surf(f)
svertka_0
3d
conv_3d
%-- 19.04.13, 0:51 --%
A = [[1 2; 1 1], [1 1; 1 1]]
A = [[1 2; 1 1]; [1 1; 1 1]]
A[:,:,1] = [1 2; 3 4]
A = [1 2; 3 4]
A[,,1 = [1 2; 3 4]
A[,,1] = [1 2; 3 4]
A = zeros([3,3,2])
A(:,:,1) = [ 1 2 3 ; 1 2 3 ; 1 2 3 ]
A(:,:,2) = [ 4 5 6 ; 4 5 6 ; 4 5 6 ]
B = reshape(A,3,[])
%-- 19.04.13, 2:22 --%
%-- 23.04.13, 11:37 --%
factor(8191)
factor(2048+1)
a = randn(8191, 29);
tic; fft(a); toc;
tic; fft(a'); toc;
tic; fft(a); toc;
tic; a = fft(a); toc;
size(a)
a = randn(8192, 29);
tic; a = fft(a); toc;
tic; a * a; toc;
b = randn(1000, 1000)
b = randn(1000, 1000);
tic; b * b; toc;
tic; [u, s, v] = svd(b); toc;
