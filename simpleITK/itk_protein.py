import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
from scipy.sparse import issparse
from skimage.color import rgb2gray
from imageio.v2 import imwrite
import cv2

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


datasets=["LUAD_2_A", "TSU_20_1",  "TSU_33",
             "LUAD_3_A", "TSU_24", "TSU_30", "TSU_35"]
root = Path("/media/huifang/data/registration/phenocycler/")
datadir = "/media/huifang/data/registration/phenocycler/H5ADs/"
start_idx = 0
for data in datasets[start_idx:]:

    print(data)
    protein_img_path = datadir +f"{data}_protein_DAPI_trans2.png"
    xenium_img_path = datadir +f"{data}_xenium_DAPI_trans2.png"
    visium_img_path = datadir +f"{data}_visium_GRAY_trans2.png"

    displacement_field, fixed_image, moving_image, warped_image = get_simpleITK_transformation(protein_img_path,
                                                                                               visium_img_path)
    print('calculation is done, perform transformations..')



    img_before_registration = cv2.addWeighted(fixed_image, 0.5, moving_image, 0.5, 0)
    blended_image = cv2.addWeighted(fixed_image, 0.5, warped_image, 0.5, 0)

    # # Create subplots
    f, a = plt.subplots(1, 2, figsize=(10, 10))
    # Show overlayed images
    a[0].imshow(img_before_registration)
    a[0].set_title("Before Registration")
    a[0].axis("off")  # Hide axes

    # Show overlayed images
    a[1].imshow(blended_image)
    a[1].set_title("After Registration")
    a[1].axis("off")  # Hide axes


    plt.tight_layout()
    plt.show()