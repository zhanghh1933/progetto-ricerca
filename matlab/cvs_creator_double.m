close all; clear all; 

addpath('jpeg/');

source = '../dataset_complete/original';
w_w = '../double_sharing/whatsapp/whatsapp';
w_m = '../double_sharing/whatsapp/messenger';
w_t = '../double_sharing/whatsapp/telegram';

m_w = '../double_sharing/messenger/whatsapp';
m_m = '../double_sharing/messenger/messenger';
m_t = '../double_sharing/messenger/telegram';

t_w = '../double_sharing/telegram/whatsapp';
t_m = '../double_sharing/telegram/messenger';
t_t = '../double_sharing/telegram/telegram';

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
folder_w_w = dir(w_w);
folder_w_m = dir(w_m);
folder_w_t = dir(w_t);

folder_m_w = dir(m_w);
folder_m_m = dir(m_m);
folder_m_t = dir(m_t);

folder_t_w = dir(t_w);
folder_t_m = dir(t_m);
folder_t_t = dir(t_t);

kk = 1;

data{1, kk} = 'name'; kk = kk + 1;

data{1, kk} = 'original_size';kk = kk + 1;
data{1, kk} = 'original_original_size';kk = kk + 1;
data{1, kk} = 'original_h';kk = kk + 1;
data{1, kk} = 'original_w';kk = kk + 1;

data{1, kk} = 'whatsapp_whatsapp_size';kk = kk + 1;
data{1, kk} = 'whatsapp_whatsapp_original_size';kk = kk + 1;
data{1, kk} = 'whatsapp_whatsapp_h';kk = kk + 1;
data{1, kk} = 'whatsapp_whatsapp_w';kk = kk + 1;
data{1, kk} = 'whatsapp_whatsapp_q';kk = kk + 1;

data{1, kk} = 'whatsapp_messenger_size';kk = kk + 1;
data{1, kk} = 'whatsapp_messenger_original_size';kk = kk + 1;
data{1, kk} = 'whatsapp_messenger_h';kk = kk + 1;
data{1, kk} = 'whatsapp_messenger_w';kk = kk + 1;
data{1, kk} = 'whatsapp_messenger_q';kk = kk + 1;

data{1, kk} = 'whatsapp_telegram_size';kk = kk + 1;
data{1, kk} = 'whatsapp_telegram_original_size';kk = kk + 1;
data{1, kk} = 'whatsapp_telegram_h';kk = kk + 1;
data{1, kk} = 'whatsapp_telegram_w';kk = kk + 1;
data{1, kk} = 'whatsapp_telegram_q';kk = kk + 1;


data{1, kk} = 'messenger_whatsapp_size';kk = kk + 1;
data{1, kk} = 'messenger_whatsapp_original_size';kk = kk + 1;
data{1, kk} = 'messenger_whatsapp_h';kk = kk + 1;
data{1, kk} = 'messenger_whatsapp_w';kk = kk + 1;
data{1, kk} = 'messenger_whatsapp_q';kk = kk + 1;

data{1, kk} = 'messenger_messenger_size';kk = kk + 1;
data{1, kk} = 'messenger_messenger_original_size';kk = kk + 1;
data{1, kk} = 'messenger_messenger_h';kk = kk + 1;
data{1, kk} = 'messenger_messenger_w';kk = kk + 1;
data{1, kk} = 'messenger_messenger_q';kk = kk + 1;

data{1, kk} = 'messenger_telegram_size';kk = kk + 1;
data{1, kk} = 'messenger_telegram_original_size';kk = kk + 1;
data{1, kk} = 'messenger_telegram_h';kk = kk + 1;
data{1, kk} = 'messenger_telegram_w';kk = kk + 1;
data{1, kk} = 'messenger_telegram_q';kk = kk + 1;


data{1, kk} = 'telegram_whatsapp_size';kk = kk + 1;
data{1, kk} = 'telegram_whatsapp_original_size';kk = kk + 1;
data{1, kk} = 'telegram_whatsapp_h';kk = kk + 1;
data{1, kk} = 'telegram_whatsapp_w';kk = kk + 1;
data{1, kk} = 'telegram_whatsapp_q';kk = kk + 1;

data{1, kk} = 'telegram_messenger_size';kk = kk + 1;
data{1, kk} = 'telegram_messenger_original_size';kk = kk + 1;
data{1, kk} = 'telegram_messenger_h';kk = kk + 1;
data{1, kk} = 'telegram_messenger_w';kk = kk + 1;
data{1, kk} = 'telegram_messenger_q';kk = kk + 1;

data{1, kk} = 'telegram_telegram_size';kk = kk + 1;
data{1, kk} = 'telegram_telegram_size';kk = kk + 1;
data{1, kk} = 'telegram_telegram_h';kk = kk + 1;
data{1, kk} = 'telegram_telegram_w';kk = kk + 1;
data{1, kk} = 'telegram_telegram_q';kk = kk + 1;


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
    
    
    
    %% whatsapp_whatsapp
    
    data{i, col} = folder_w_w(f).bytes; col = col + 1;
    im_w = imread([w_w '/' folder_w_w(f).name]);
    
    imwrite(im_w , 'temp\test.jpg', 'Quality', 100);
    im_w_size = dir('temp\test.jpg');
    data{i, col} = im_w_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_w);
    
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([w_w '/' folder_w_w(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% whatsapp_messenger
    
    data{i, col} = folder_w_m(f).bytes; col = col + 1;
    im_w = imread([w_m '/' folder_w_m(f).name]);
    
    imwrite(im_w , 'temp\test.jpg', 'Quality', 100);
    im_w_size = dir('temp\test.jpg');
    data{i, col} = im_w_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_w);
    
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([w_m '/' folder_w_m(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% whatsapp_telegram
    
    data{i, col} = folder_w_t(f).bytes; col = col + 1;
    im_w = imread([w_t '/' folder_w_t(f).name]);
    
    imwrite(im_w , 'temp\test.jpg', 'Quality', 100);
    im_w_size = dir('temp\test.jpg');
    data{i, col} = im_w_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_w);
    
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([w_t '/' folder_w_t(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    
    %% telegram_whatsapp
    
    data{i, col} = folder_t_w(f).bytes; col = col + 1;
    im_t = imread([t_w '/' folder_t_w(f).name]);
    
    imwrite(im_t , 'temp\test.jpg', 'Quality', 100);
    im_t_size = dir('temp\test.jpg');
    data{i, col} = im_t_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_t);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([t_w '/' folder_t_w(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% telegram_messenger
    
    data{i, col} = folder_t_m(f).bytes; col = col + 1;
    im_t = imread([t_m '/' folder_t_m(f).name]);
    
    imwrite(im_t , 'temp\test.jpg', 'Quality', 100);
    im_t_size = dir('temp\test.jpg');
    data{i, col} = im_t_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_t);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([t_m '/' folder_t_m(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% telegram_telegram
    
    data{i, col} = folder_t_t(f).bytes; col = col + 1;
    im_t = imread([t_t '/' folder_t_t(f).name]);
    
    imwrite(im_t , 'temp\test.jpg', 'Quality', 100);
    im_t_size = dir('temp\test.jpg');
    data{i, col} = im_t_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_t);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([t_t '/' folder_t_t(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
   
    
    %% messenger_whatsapp
    
    data{i, col} = folder_m_w(f).bytes; col = col + 1;
    im_m = imread([m_w '/' folder_m_w(f).name]);
    
    imwrite(im_m , 'temp\test.jpg', 'Quality', 100);
    im_m_size = dir('temp\test.jpg');
    data{i, col} = im_m_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_m);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([m_w '/' folder_m_w(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% messenger_messenger
    
    data{i, col} = folder_m_m(f).bytes; col = col + 1;
    im_m = imread([m_m '/' folder_m_m(f).name]);
    
    imwrite(im_m , 'temp\test.jpg', 'Quality', 100);
    im_m_size = dir('temp\test.jpg');
    data{i, col} = im_m_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_m);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([m_m '/' folder_m_m(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
    %% messenger_telegram
    
    data{i, col} = folder_m_t(f).bytes; col = col + 1;
    im_m = imread([m_t '/' folder_m_t(f).name]);
    
    imwrite(im_m , 'temp\test.jpg', 'Quality', 100);
    im_m_size = dir('temp\test.jpg');
    data{i, col} = im_m_size.bytes; col = col + 1 ;
    
    [x, y, z] = size(im_m);
    
    if is_x && x > y  
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    elseif is_x && x < y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    elseif ~is_x && x > y
        data{i, col} = y; col = col + 1;
        data{i, col} = x; col = col + 1;
    else
        data{i, col} = x; col = col + 1;
        data{i, col} = y; col = col + 1;
    end
    
    jpeg_im = jpeg_read([m_t '/' folder_m_t(f).name]);
    
    est = quality_calc(jpeg_im, std_Q1, std_Q2);
   
    data{i, col} = est; col = col + 1 ;
    
 
    
end


xlswrite('result_complete_double1.xls', data);
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


