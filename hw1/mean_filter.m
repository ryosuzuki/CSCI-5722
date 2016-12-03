function [ new_img ] = mean_filter(current_img, kernel_size)

  current_img = double(current_img) / 255;
  new_img = zeros(size(current_img));

  pad_x = floor(kernel_size/2)
  pad_y = floor(kernel_size/2)
  pad_img = padarray(current_img, [pad_x, pad_y])
  [n_x, n_y, n_c] = size(pad_img)

  for c = 1 : n_c
    for x = 1 + pad_x : n_x - pad_x
      for y = 1 + pad_y : n_y - pad_y
        sub_img = pad_img(x-pad_x : x+pad_x, y-pad_y : y+pad_y, c);
        new_img(x-pad_x, y-pad_y, c) = sum(sub_img) / (kernel_size*kernel_size);
      end
    end
  end

end