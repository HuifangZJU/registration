import tifffile
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
#Image.MAX_IMAGE_PIXELS = 1000000000 
Image.MAX_IMAGE_PIXELS = None 
def normImg(img):
    #avoid zeros
    vmax = img.max()
    if vmax == 0:
        return img
    else:
        img[img > vmax] = vmax
        img = img * (254 / vmax)
        img[img > 254] = 254
        return img

def save_img(tiff_image, v_start_pixel,v_end_pixel, u_start_pixel,u_end_pixel, fileid, saveformat):
    crop_img = tiff_image[v_start_pixel:v_end_pixel, u_start_pixel:u_end_pixel]
    des_dir='./'
    if saveformat == 'tif':
        des_file = des_dir + str(fileid) + '_' +str(v_start_pixel) + '_' + str(u_start_pixel) + '.tif'
        tifffile.imwrite(des_file, crop_img)
    if saveformat == 'png':
        des_file = des_dir + str(fileid) + '_' + str(v_start_pixel) + '_' + str(u_start_pixel) + '2.png'
        crop_img = normImg(crop_img)
        # plt.imshow(crop_img)
        # plt.show()
        crop_img = crop_img.astype(np.uint16)
        crop_img = Image.fromarray(crop_img)
        crop_img = crop_img.convert('L')
        crop_img.save(des_file)

#img = plt.imread('/home/huifang/workspace/data/cell_segmentation/DAPI/DAPI_09-0161_ISAP12186104_20220601.tif')
img = plt.imread('/media/huifang/data/vizgen/HumanBreastCancerPatient1/images/mosaic_DAPI_z0.tif')
print(img.shape)
test = input()
centerx = 47000
centery = 55000
img = img[centerx:,centery:]
plt.imshow(img,cmap='gray')
plt.show()


img = plt.imread('/media/huifang/data/vizgen/HumanBreastCancerPatient1/images/z0_fov_images/DAPI/masks/mosaic_DAPI_z0_fov27_cp_masks.png')
img45 = plt.imread('/media/huifang/data/vizgen/HumanBreastCancerPatient1/images/z0_fov_images/DAPI/masks/45mosaic_DAPI_z0_fov27_cp_masks.png')
f,a = plt.subplots(1,2)
a[0].imshow(img)
a[1].imshow(img45)
plt.show()

#print(img.shape)
rescaled_img = plt.imread('/home/huifang/workspace/data/cell_segmentation/DAPI_rescaled/DAPI_05-0126_ISAP12186106_20220601.tif')
save_img(rescaled_img,14400,15400,9400,10400,0,'tif')
print('done')
test = input()
print(rescaled_img.shape)
img = img[9400:10400,12000:13600]
rescaled_img = rescaled_img[14400:15400,9400:10400]
#plt.imshow(rescaled_img)
f,a = plt.subplots(1,2)
a[0].imshow(img,cmap='gray')
a[1].imshow(rescaled_img,cmap='gray')
plt.show()
