import cv2
import numpy as np
import os
from PIL import Image
# np.set_printoptions(threshold=np.inf)
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def one_width_corrosion(img):
    img = np.asarray(img)
    # print(img.max())
    # print(img.min())

    # 骨架化
    skeleton = skeletonize(img//255)
    # plt.imshow(skeleton, cmap='gray')
    # plt.show()


    kernel3 = np.ones((3, 3), np.uint8)
    onepixel_corrosion = cv2.erode(img, kernel3, iterations=1)
    onepixel_corrosion = cv2.morphologyEx(onepixel_corrosion, cv2.MORPH_CLOSE, kernel3)
    # plt.imshow(onepixel_corrosion, cmap='gray')
    # plt.show()

    out = onepixel_corrosion + skeleton*255
    out[out>=255] = 255
    # plt.imshow(out, cmap='gray')
    # plt.show()
    # cv2.imwrite('corrosion.PNG', out)

    return out

def one_width_dilation(img):
    img = np.asarray(img)

    # 骨架化
    skeleton = skeletonize(img//255)

    # 计算骨架中每个像素到最近的血管的距离
    dist_transform = distance_transform_edt(img//255)

    onepixel = np.zeros((584,565))

    # for i in range(1, 584-1):
    #     for j in range(1, 565-1):
    #         if skeleton[i][j]==1 and dist_transform[i][j] == 1:
    #             onepixel[i][j] = 255

    for i in range(1, 584-1):
        for j in range(1, 565-1):
            one_p = True
            if skeleton[i][j]==1:   #如果在血管中心线上
                for a in [-1,0,1]:   #查找附近9个像素
                    if one_p==False:
                        break
                    for b in [-1,0,1]:
                        if dist_transform[i+a][j+b] > 1:     #如果附近有大于1的距离，则删除这个点
                            one_p = False
                            break
                if one_p==True:
                    onepixel[i][j] = 255

    kernel3 = np.ones((3, 3), np.uint8)
    onepixel_dilation = cv2.dilate(onepixel, kernel3, iterations=1)
    # onepixel_dilation = cv2.morphologyEx(onepixel, cv2.MORPH_OPEN, kernel3)

    out = onepixel_dilation + img
    out[out>255] = 255

    return out


def dilation_median(img):
    img_arr = np.asarray(img)

    # 先定义一个核大小
    kernel1 = np.ones((1, 1), np.uint8)
    kernel2 = np.ones((2, 2), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)


    # Opening operation 执行开操作 去除很细的边缘
    img_dila2 = cv2.morphologyEx(img_arr, cv2.MORPH_OPEN, kernel3)

    # Obtain edges    获得边   对边像素执行必操作，连通断裂的边缘
    onepixel = img_arr - img_dila2
    onepixel = cv2.morphologyEx(onepixel, cv2.MORPH_CLOSE, kernel1)

    # Dilation of the repaired edges 膨胀边缘
    onepixel_dilation = cv2.dilate(onepixel.astype(np.uint8), kernel2, iterations=1)

    # Combine original image and dilated edges   将获得的边与原图相加   得到自适应膨胀血管
    out = img_arr + onepixel_dilation

    return out


# Input and output directories
input_folder = '/root/FR-UNet-master/dataset/DRIVE/training/1st_manual'
output_folder = '/root/FR-UNet-master/dataset/DRIVE/training/1st_manual_corrosion'
# output_folder1 = 'D:/pythonProject/2022/FR-UNet-master/auto_middle_vessel/DRIVE_origin'


# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each .gif file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.gif'):
        filepath = os.path.join(input_folder, filename)
        # print(filepath)

        # Read the gif image
        # img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        img = Image.open(filepath)
        # print("origin ==",np.asarray(img))

        # Apply dilation_median function
        # processed_img = one_width_dilation(img)
        processed_img = one_width_corrosion(img)
        processed_img[processed_img > 0] = 255
        # print("dilation ==",np.asarray(processed_img))

        # Write the processed image to the output folder
        output_filepath = os.path.join(output_folder, filename[:-3]+"PNG")
        print(output_filepath)
        cv2.imwrite(output_filepath, processed_img)

        # output_filepath = os.path.join(output_folder1, filename[:-3] + "PNG")
        # img.save(output_filepath)

        # break




print("Processing completed!")
