close all; clear all; 

addpath('jpeg/');

source = '../dataset_complete/original';
w_w_what = '../dataset_complete/whatsapp';
w_w_mess = '../dataset_complete/messenger';
w_w_tele = '../dataset_complete/telegram';

std_Q1 = [
		16,  11,  10,  16,  24,  40,  51,  61;
  		12,  12,  14,  19,  26,  58,  60,  55;
 	 	14,  13,  16,  24,  40,  57,  69,  56;
  		14,  17,  22,  29,  51,  87,  80,  62;
  		18,  22,  37,  56,  68, 109, 103,  77;
  		24,  35,  55,  64,  81, 104, 113,  92;
  		49,  64,  78,  87, 103, 121, 120, 101;
  		72,  92,  95,  98, 112, 100, 103,  99 ];
    
std_Q2 = [
		17,  18,  24,  47,  99,  99,  99,  99;
  		18,  21,  26,  66,  99,  99,  99,  99;
  		24,  26,  56,  99,  99,  99,  99,  99;
  		47,  66,  99,  99,  99,  99,  99,  99;
  		99,  99,  99,  99,  99,  99,  99,  99;
  		99,  99,  99,  99,  99,  99,  99,  99;
  		99,  99,  99,  99,  99,  99,  99,  99;
  		99,  99,  99,  99,  99,  99,  99,  99 ];


folder_original = dir(source);
folder_w_w_what = dir(w_w_what);
folder_w_w_mess = dir(w_w_mess);
folder_w_w_tele = dir(w_w_tele);

g = 1;

data{1, g} = 'name'; g = g+1;

data{1, g} = 'original_size'; g = g+1;
data{1, g} = 'real_size'; g = g+1;
data{1, g} = 'original_h';g = g+1;
data{1, g} = 'original_w';g = g+1;

data{1, g} = 'whatsapp_size';g = g+1;
data{1, g} = 'whatsapp_original_size';g = g+1;
data{1, g} = 'whatsapp_h';g = g+1;
data{1, g} = 'whatsapp_w';g = g+1;
data{1, g} = 'whatsapp_q';g = g+1;

data{1, g} = 'messenger_size';g = g+1;
data{1, g} = 'messenger_original_size';g = g+1;
data{1, g} = 'messenger_h';g = g+1;
data{1, g} = 'messenger_w';g = g+1;
data{1, g} = 'messenger_q';g = g+1;

data{1, g} = 'telegram_size';g = g+1;
data{1, g} = 'telegram_original_size';g = g+1;
data{1, g} = 'telegram_h';g = g+1;
data{1, g} = 'telegram_w';g = g+1;
data{1, g} = 'telegram_q';g = g+1;



for f = 3:numel(folder_original)
    col = 1;
    i = f - 1;
    is_x = false;
    
    name = strsplit(folder_original(f).name, '.');
    name = name{1};
    name = strsplit(name, '-');
    last = numel(name);
    name = name{last};
    data{i, col} = name; col = col + 1;
    
    %% orignal image
    
    
    data{i, col} = folder_original(f).bytes; col = col + 1;
    im_o = imread([source '/' folder_original(f).name]);
    
    imwrite(im_o , 'temp\test.jpg', 'Quality', 100);
    im_or_size = dir('temp\test.jpg');
    data{i, col} = im_or_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_o);
    
    if x > y
        is_x = true;
    end
    
    data{i, col} = x; col = col + 1;
    data{i, col} = y; col = col + 1;
    

    
%     %% whatsapp
%     
%     data{i, col} = folder_w_w_what(f).bytes; col = col + 1;
%     im_w = imread([w_w_what '/' folder_w_w_what(f).name]);
%     
%     imwrite(im_w , 'temp\test.jpg', 'Quality', 100);
%     im_w_size = dir('temp\test.jpg');
%     data{i, col} = im_w_size.bytes; col = col + 1 ;
%     
%     [x, y, z] = size(im_w);
%     
%     
%     if is_x && x > y  
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     elseif is_x && x < y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     elseif ~is_x && x > y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     else
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     end
%     
%     jpeg_im = jpeg_read([w_w_what '/' folder_w_w_what(f).name]);
%     
%     est = quality_calc(jpeg_im, std_Q1, std_Q2);
%    
%     data{i, col} = est; col = col + 1 ;
%     
%     %% messenger
%     
%     data{i, col} = folder_w_w_mess(f).bytes; col = col + 1;
%     im_m = imread([w_w_mess '/' folder_w_w_mess(f).name]);
%     
%     imwrite(im_m , 'temp\test.jpg', 'Quality', 100);
%     im_m_size = dir('temp\test.jpg');
%     data{i, col} = im_m_size.bytes; col = col + 1 ;
%     
%     [x, y, z] = size(im_m);
%     
%     if is_x && x > y  
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     elseif is_x && x < y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     elseif ~is_x && x > y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     else
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     end
%     
%     jpeg_im = jpeg_read([w_w_mess '/' folder_w_w_mess(f).name]);
%     
%     est = quality_calc(jpeg_im, std_Q1, std_Q2);
%    
%     data{i, col} = est; col = col + 1 ;
%     
%     %% telegram
%     
%     data{i, col} = folder_w_w_tele(f).bytes; col = col + 1;
%     im_t = imread([w_w_tele '/' folder_w_w_tele(f).name]);
%     
%     imwrite(im_t , 'temp\test.jpg', 'Quality', 100);
%     im_t_size = dir('temp\test.jpg');
%     data{i, col} = im_t_size.bytes; col = col + 1 ;
%     
%     [x, y, z] = size(im_t);
%     
%     if is_x && x > y  
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     elseif is_x && x < y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     elseif ~is_x && x > y
%         data{i, col} = y; col = col + 1;
%         data{i, col} = x; col = col + 1;
%     else
%         data{i, col} = x; col = col + 1;
%         data{i, col} = y; col = col + 1;
%     end
%     
%     jpeg_im = jpeg_read([w_w_tele '/' folder_w_w_tele(f).name]);
%     
%     est = quality_calc(jpeg_im, std_Q1, std_Q2);
%    
%     data{i, col} = est; col = col + 1 ;
    
end


xlswrite('result_complete_with_original_size.xls', data);
function quality = quality_calc(info, std_Q1, std_Q2)



    q1 = info.quant_tables{1,1};
    q1 = q1 ./ std_Q1;
    s1 = sum(q1);
    s1 = sum(s1);
    sc_quality1 = s1 / 64 ;
    sc_quality1 = sc_quality1 * 100;
    
    if sc_quality1 <= 100
        est_quality_1 = (200 - sc_quality1) / 2 ;
    else
        est_quality_1 = 5000 / sc_quality1 ;
    end
        
    
    q2 = info.quant_tables{1,2};
    q2 = q2 ./ std_Q2;
    s2 = sum(q2);
    s2 = sum(s2);
    sc_quality2 = s2 / 64 ;
    sc_quality2 = sc_quality2 * 100;
    
    if sc_quality2 <= 100
        est_quality_2 = (200 - sc_quality2) / 2 ;
    else
        est_quality_2 = 5000 / sc_quality2 ;
    end


    est_quality = (est_quality_1 + est_quality_2) / 2;
    quality = est_quality;

end


