from code import *


import Augmentor

p = Augmentor.Pipeline(r"C:\Users\Manh\Desktop\TH\Output\handwritten")
p.rotate90(probability=0.4)
p.rotate270(probability=0.4)
p.flip_left_right(probability=0.7)
p.flip_top_bottom (probability=0.3)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.crop_random (probability=0.5, percentage_area=0.8)
p.resize(probability=0.2, width=1200, height=900)
p.sample(10)

# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\A_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\B_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\C_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\D_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\E_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\F_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\G_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\H_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\I_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\J_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\K_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\L_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\M_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\N_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\O_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\P_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\Q_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\R_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\S_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\T_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\U_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\V_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\X_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\Y_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\Z_hoa",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\W_hoa",r"C:\Users\Manh\Desktop\TH\key_")


# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\a_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\b_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\c_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\d_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\e_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\f_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\g_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\h_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\i_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\j_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\k_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\l_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\m_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\n_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\o_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\p_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\q_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\r_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\s_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\t_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\u_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\v_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\x_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\y_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\z_thuong",r"C:\Users\Manh\Desktop\TH\key_")
# save_extracted_features(r"C:\Users\Manh\Desktop\TH\data\w_thuong",r"C:\Users\Manh\Desktop\TH\key_")

# convert_parquet('4g', r"C:\Users\Manh\Desktop\TH", r"C:\Users\Manh\Desktop\TH\key_", "database")

# cal_centers('4g', r"C:\Users\Manh\Desktop\TH", 'database', 'center', 10)

# cal_encodes_tranningset(r"C:\Users\Manh\Desktop\TH\data", r"C:\Users\Manh\Desktop\TH\center.npy");
# 

most_similar_image_demo_display(r"C:\Users\Manh\Desktop\TH\Input\img058-055.png", r"C:\Users\Manh\Desktop\TH\data", r"C:\Users\Manh\Desktop\TH\center.npy", ratio=1)

# list_img = os.listdir(r"C:\Users\Manh\Desktop\TH\Input")
# for image_name in list_img:
# 	image_path = os.path.join(r"C:\Users\Manh\Desktop\TH\Input", image_name) 
# 	most_similar_image_demo_display(image_path, r"C:\Users\Manh\Desktop\TH\data", r"C:\Users\Manh\Desktop\TH\center.npy", ratio=1)






