function displayImage(current_img, new_img, name)
  figure
  subplot(1, 2, 1);
  imagesc(current_img);
  subplot(1, 2, 2);
  imagesc(new_img);
  imwrite(new_img, strjoin({name, 'jpg'}, '.') )
end
