import os
import cv2
import numpy as np
import imageio.v2 as imageio
from skimage.morphology import skeletonize
import random

def process_images(path_images,path_1st_manual,path_images_processed,patch_size,dataset_type):
    for name_images in os.listdir(path_images):
        if name_images.endswith(".tif"):  # 假设处理.tif格式的图像
            path_to_images = os.path.join(path_images, name_images)
            image = imageio.imread(path_to_images)
            GT = imageio.imread(path_1st_manual+'/'+name_images.replace("_{}.tif".format(dataset_type), "_manual1.gif"))  # 根据文件名规则获取对应的GT图像

            if len(image.shape) > 2:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image

            patches = []
            for i in range(0, image_gray.shape[0], patch_size):
                for j in range(0, image_gray.shape[1], patch_size):
                    patch = image_gray[i:i+patch_size, j:j+patch_size]
                    if patch.shape == (patch_size, patch_size):
                        patches.append((i, j, patch))

            selected_indices = random.sample(range(len(patches)), int(0.6 * len(patches)))

            masked_image = np.zeros_like(image_gray)
            processed_image = np.zeros_like(image_gray)
            for idx, (i, j, patch) in enumerate(patches):
                if idx in selected_indices:         #打掩膜
                    GT_patch = GT[i:i+patch_size, j:j+patch_size] #对应的GT_patch
                    thinned_img = skeletonize(GT_patch // 255) * 255
                    thinned_img = thinned_img.astype(np.uint8)
                    kernel_dila = np.ones((9, 9), np.uint8)
                    dila_img = cv2.dilate(thinned_img, kernel_dila, iterations=1)
                    # mins_img = dila_img - thinned_img     #这就是掩膜
                    mins_img = dila_img     #这就是掩膜   不扣除中心线
                    edge_patch = cv2.subtract(patch, (mins_img // 255) * patch)
                    processed_patch = edge_patch       
                else:
                    processed_patch = patch        #这里面是没有掩膜的
                    mins_img = np.zeros_like(patch)

                processed_image[i:i+patch_size, j:j+patch_size] = processed_patch
                masked_image[i:i+patch_size, j:j+patch_size] = mins_img // 255

            # 保存结果图像
            processed_filepath = os.path.join(path_images_processed,name_images.replace(".tif", ".PNG"))            
            grey_filepath = os.path.join('/root/FR-UNet-master/dataset/DRIVE', dataset_type, 'images_grey', name_images.replace(".tif", ".PNG"))            
            masked_imagefile = os.path.join('/root/FR-UNet-master/dataset/DRIVE', dataset_type, 'masked_image', name_images.replace(".tif", ".PNG"))
            
            print(processed_filepath)
            cv2.imwrite(processed_filepath, processed_image)
            cv2.imwrite(grey_filepath, image_gray)
            cv2.imwrite(masked_imagefile, masked_image)
            
            
data_type = "training"
# 调用函数处理指定文件夹中的所有图片
path_images = os.path.join('/root/FR-UNet-master/dataset/DRIVE', data_type, 'images')
path_1st_manual = os.path.join('/root/FR-UNet-master/dataset/DRIVE', data_type, '1st_manual')
path_images_processed = os.path.join('/root/FR-UNet-master/dataset/DRIVE', data_type, 'images_processed')
patch_size = 48
process_images(path_images,path_1st_manual,path_images_processed,patch_size,data_type)
