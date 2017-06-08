% 
% Deceived Non Local Means Filter using Integral Images and the  
% Fast Fourier Transform
%  
% Parameters:   
%  I = Input image
%  w = Search window anchor size
%  w_n = Neighborhood anchor size
%  sigma_r = gaussian strength
% 


function fastNLMFilter(I, w, w_n, sigma_r)

  G = rgb2gray( I );
  G = double(G);
  [size_x, size_y] = size(G);
  
  R = zeros(size_x, size_y);
  
  II2 = integralImage(G.^2);
  
  %Computing Gaussian domain weigths
  sigma_s = w/1.5;
  [X, Y] = meshgrid(-w:w, -w:w);
  X = power(X,2);
  Y = power(Y,2);
  S = X+Y;
  S = S/(-2*power(sigma_s,2));
  GaussW= exp(S);
  
  U = NoAdaptativeUSM(G, 3, 17, 0.005); 
  
  parfor i = 1:size_x
    for j = 1:size_y
    
    if ((i>w+w_n) && (j>w+w_n) && (i<size_x-w-w_n) && (j<size_y-w-w_n))
      %Extract local 
      iMin = max(i - w - w_n , 1);
      iMax = min(i + w + w_n, size_x);
      jMin = max(j - w - w_n, 1);
      jMax = min(j + w + w_n, size_y);
      %Get current window
      I = G(iMin:iMax,jMin:jMax);
      [sizeW_x, sizeW_y] = size(I);
      %Create output matrix
      O = zeros(sizeW_x-2*w_n,sizeW_y-2*w_n);
      
      %Extract pixel neighborhood P local region.
      mMin_p = i - w_n;
      mMax_p = i + w_n;
      nMin_p = j - w_n;
      nMax_p = j + w_n;
      
      %Get sum of squad neighborhood P
      sum_p = II2(mMin_p, nMin_p) + II2(mMax_p+1,nMax_p+1) - II2(mMin_p,nMax_p+1) - II2(mMax_p+1,nMin_p);
      
      %Get current neighborhood P
      neighbor_p = G(mMin_p:mMax_p, nMin_p:nMax_p);
      [sizeP_x, sizeP_y] = size(neighbor_p);
      
      %Perform correlation
      % output size 
      mm = sizeW_x + sizeP_x - 1;
      nn = sizeW_y + sizeP_y - 1;
      % pad, multiply and transform back
      C = real(ifft2(fft2(I,mm,nn).* fft2(rot90(neighbor_p,2),mm,nn)));
      % padding constants (for output of size == size(A))
      padC_m = ceil((sizeP_x-1)./2);
      padC_n = ceil((sizeP_y-1)./2);
      % convolution result
      correlation = C(padC_m+1:sizeW_x+padC_m, padC_n+1:sizeW_y+padC_n);
      
      for m = 1+w_n:sizeW_x-w_n
        mMin_w = iMin + m-1 - w_n;
        mMax_w = iMin + m-1 + w_n;
        
        for n = 1+w_n:sizeW_y-w_n
          nMin_w = jMin + n-1 - w_n;
          nMax_w = jMin + n-1 + w_n;
          sum_w= II2(mMin_w, nMin_w) + II2(mMax_w+1,nMax_w+1) - II2(mMin_w,nMax_w+1) - II2(mMax_w+1,nMin_w);          
          O(m-w_n,n-w_n)= sum_p + sum_w -2*correlation(m,n);
        end
      end
      O = exp(O/(-2* sigma_r^2));
      O = O.*GaussW;
      norm_factor = sum(sum(O));      
      R(i,j)= sum(sum(O.*U(iMin+w_n:iMax-w_n,jMin+w_n:jMax-w_n)))/norm_factor;
    end
   end
  end
   imshow(R, []);
end

%No adaptative lapacian
function U = NoAdaptativeUSM(SrcImage, lambda, kernelSize, kernelSigma)
    kernel = fspecial('log',kernelSize,kernelSigma);
    Z = imfilter(SrcImage,-kernel);
    maxZ= max(max(abs(Z)));
    maxSrc= max(max(SrcImage));
    Z = maxSrc * (Z / maxZ);
    U = SrcImage + lambda * Z;
end