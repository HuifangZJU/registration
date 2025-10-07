import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def run_simpleITK(fixed_image,moving_image):
    # Set up the B-spline transform
    transform = sitk.BSplineTransformInitializer(fixed_image, [3, 3], order=3)  # Control points grid size

    # Registration setup
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(transform)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-3, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform


def get_simpleITK_transformation(fixed_path, moving_path):
    # Read original-size images
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    # 2) Perform registration on the small images
    itk_transform = run_simpleITK(fixed_image, moving_image)  # your registration method

    # 3) Generate the displacement field in the original full-size domain
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(fixed_image)  # <-- reference is the full-size fixed image
    displacement_field = displacement_filter.Execute(itk_transform)

    # 4) Resample the original moving image at full resolution
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)  # use the full-size fixed as reference
    resampler.SetTransform(itk_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered_image = resampler.Execute(moving_image)

    # Convert images to numpy arrays, if needed
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    registered_array = sitk.GetArrayViewFromImage(registered_image)

    # Normalize for display
    fixed_norm = normalize(fixed_array)
    moving_norm = normalize(moving_array)
    registered_norm = normalize(registered_array)
    return displacement_field, fixed_norm, moving_norm, registered_norm


def transform_uv_with_displacement(uv_coords, deformation_field):
    deformation_np = sitk.GetArrayFromImage(deformation_field)
    deformation_size = deformation_field.GetSize()
    transformed_coords = []
    for u, v in uv_coords:
        u_int, v_int = int(u), int(v)

        # Ensure UV coordinates are within bounds
        if 0 <= u_int < deformation_size[0] and 0 <= v_int < deformation_size[1]:
            # Sample displacement at (u, v)
            displacement = deformation_np[v_int,u_int]  # (v, u) - numpy row-major

            # Apply displacement directly to UV
            u_transformed = u - displacement[0]  # x-component
            v_transformed = v - displacement[1]  # y-component
            transformed_coords.append([u_transformed, v_transformed])
        else:
            # If out of bounds, keep original point
            transformed_coords.append([u, v])
    return np.array(transformed_coords)




import os
root = "/media/huifang/data/registration"
datalist = "imglist/scc_imglist/pairwise_align_test.txt"
datapath = os.path.join(root, datalist)
f = open(datapath, 'r')
lines = f.readlines()
prefix=['2_0','2_1','5_0','5_1','9_0','9_1','10_0','10_1']
for i in range(len(lines)):
    i=7
    data_pair = lines[i].strip()
    data_pair = data_pair.split(' ')
    fixed_image_path = root + data_pair[0] + '_image_512.png'
    moving_image_path = root + data_pair[1] + '_image_512.png'
    displacement_field,fixed_image,moving_image, warped_image = get_simpleITK_transformation(fixed_image_path,moving_image_path)
    print('calculation is done, perform transformations..')

    fixed_val = np.load(root + data_pair[0] + '_validation.npz')
    fixed_pts = fixed_val["coord"]
    fixed_label = fixed_val["label"]

    moving_val = np.load(root + data_pair[1] + '_validation.npz')
    moving_pts = moving_val["coord"]
    moving_label = moving_val["label"]

    # warped_pts = moving_pts
    warped_pts = transform_uv_with_displacement(moving_pts, displacement_field)
    #
    # np.savez("/media/huifang/data/registration/result/pairwise_align/SCC/simpleitk/" + prefix[i] + "_result", pts1=fixed_pts,pts2=warped_pts,img1=fixed_image,img2=warped_image,label1=fixed_label,label2=moving_label)
    # np.savez("/media/huifang/data/registration/result/pairwise_align/SCC/initial/" + prefix[i] + "_result", pts1=fixed_pts, pts2=moving_pts,img1=fixed_image, img2=moving_image, label1=fixed_label, label2=moving_label)




    f, a = plt.subplots(1, 3, figsize=(10, 5))
    # Show warped image 1 with transformed points
    a[0].imshow(fixed_image)

    a[0].scatter(fixed_pts[:, 0], fixed_pts[:, 1], c=fixed_label, cmap='tab10', s=5,
                    label="Slice 1")
    a[0].set_title("Image 1 with Original Points")
    a[0].legend()

    # Show warped image 2 with transformed points
    a[1].imshow(moving_image)
    a[1].scatter(moving_pts[:, 0], moving_pts[:, 1],c=moving_label, cmap='tab10', s=5,
                    label="Slice 2")
    a[1].set_title("Image 2 with Original Points")
    a[1].legend()


    # Show warped image 2 with transformed points
    a[2].imshow(warped_image)
    a[2].scatter(warped_pts[:, 0], warped_pts[:, 1], c=moving_label, cmap='tab10', s=5,
                 label="Adjusted Warped Slice 2")
    a[2].set_title("Warped Image 2 with Transformed Points")
    a[2].legend()

    plt.show()
    # #
    # #
    # #
    # #
    # #
    # # Overlay images
    img_before_registration = cv2.addWeighted(fixed_image, 0.5, moving_image, 0.5, 0)
    blended_image = cv2.addWeighted(fixed_image, 0.5, warped_image, 0.5, 0)

    # Create subplots
    f, a = plt.subplots(2, 2, figsize=(10, 10))

    # Scatter plot of original points
    a[0, 0].scatter(fixed_pts[:, 0], fixed_pts[:, 1], c=fixed_label, cmap='tab10',label="Slice 1", alpha=0.6)
    a[0, 0].scatter(moving_pts[:, 0], moving_pts[:, 1], c=moving_label, cmap='tab10',label="Slice 2", alpha=0.6)
    a[0, 0].invert_yaxis()  # Match image coordinate system
    a[0, 0].set_title("Original Points")
    a[0, 0].set_aspect('equal')
    a[0, 0].legend()

    # Scatter plot of warped (transformed) points
    a[0, 1].scatter(fixed_pts[:, 0], fixed_pts[:, 1],c=fixed_label, cmap='tab10', label="Warped Slice 1", alpha=0.6)
    a[0, 1].scatter(warped_pts[:, 0], warped_pts[:, 1],c=moving_label, cmap='tab10', label="Warped Slice 2", alpha=0.6)
    a[0, 1].invert_yaxis()
    a[0, 1].set_title("Warped Points")
    a[0, 1].set_aspect('equal')
    a[0, 1].legend()
    # Show overlayed images
    a[1, 0].imshow(img_before_registration)
    a[1, 0].set_title("Before Registration")
    a[1, 0].axis("off")  # Hide axes

    # Show overlayed images
    a[1, 1].imshow(blended_image)
    a[1, 1].set_title("After Registration")
    a[1, 1].axis("off")  # Hide axes


    plt.tight_layout()
    plt.show()

