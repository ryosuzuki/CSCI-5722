%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ryo Suzuki
% rysu7393
% 105790212
% ryo.suzuki@colorado.edu
%
% CSCI-5722 Computer Vision
% Ioana Fleming
% Homework Assignment 4
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [c, b, dist, str] = lcs(X,Y)
  n = length(X);
  m = length(Y);
  c = zeros(n+1, m+1);
  c(1,:) = 0;
  c(:,1) = 0;
  b = zeros(n+1, m+1);
  b(:,1) = 1;
  b(1,:) = 2;

  for i = 2:n+1
    for j = 2:m+1
      if (X(i-1) == Y(j-1))
        c(i,j) = c(i-1,j-1) + 1;
        b(i,j) = 3; % up and left
      elseif (c(i-1,j) >= c(i,j-1))
        c(i,j) = c(i-1,j);
        b(i,j) = 1; % up
      else
        c(i,j) = c(i,j-1);
        b(i,j) = 2; % left
      end
    end
  end
  c(:,1) = [];
  c(1,:) = [];
  b(:,1) = [];
  b(1,:) = [];
  dist = c(n,m);

  D = (dist / min(m,n));
  if (dist == 0)
    str = '';
  else
    i = n;
    j = m;
    p = dist;
    str = {};
    while(i>0 && j>0)
      if(b(i,j) == 3)
        str{p} = X(i);
        p = p-1;
        i = i-1;
        j = j-1;
      elseif(b(i,j) == 1)
        i = i-1;
      elseif(b(i,j) == 2)
        j = j-1;
      end
    end

    if ischar(str{1})
      str = char(str)';
    else
      str = cell2mat(str);
    end
  end
end
