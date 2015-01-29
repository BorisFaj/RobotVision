function visualize3DInfo(depth, R, G, B,ref1,ref2,showSurface)
  
C(:,:,1)=double(R);
C(:,:,2)=double(G);
C(:,:,3)=double(B);
D =  1- depth/8;

% C and D values are between 0 and 1
% Depth values are between 0 and 8

if(showSurface)
    
    figure(ref1)
    clf(ref1)
    surface(1:640,1:480,depth,C,'FaceColor','texturemap','EdgeColor','none');
    axis off;
    campos([320,-2000,0])
    
end

figure(ref2)
clf(ref2)

subplot(2,2,1)
imshow(C)
axis([0 640 0 480]);
title('Visual Image')

subplot(2,2,2)
imshow(D)
axis([0 640 0 480]);
title('Depth Image')

subplot(2,2,3)

img1=rgb2gray(C);
grayImg=mat2gray(img1);    
a = imhist(grayImg);     
a = a/sum(a);       
bar(a)
axis([0 255 0 max(a)+0.05]);
title('Grayscale histogram')

subplot(2,2,4)
histA = hist(depth,0.5:1:8);
histB = histA/sum(histA);
bar(histB);
axis([0.5 8.5 0 max(histB)+0.1]);
title('Depth histogram')

clear('points3d')
clear('C')
clear('D')
clear('depth')
    