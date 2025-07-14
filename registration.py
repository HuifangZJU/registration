import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def resize_to_target(image, target_size):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Compute new spacing to preserve physical dimensions
    new_spacing = [
        original_spacing[0] * (original_size[0] / target_size[0]),
        original_spacing[1] * (original_size[1] / target_size[1])
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# Normalize images to [0, 1] for overlay
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# Create elegant overlays (Cyan for Moving, Magenta for Fixed)
def create_overlay(fixed, moving):
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay

imagelist = "/home/huifang/workspace/data/imagelists/registration_human_mouse.txt"
images = open(imagelist)
lines = images.readlines()

for i,line in enumerate(lines):
    # fixed_path, moving_path = line.split()

    # fixed_path = '/media/huifang/data/fiducial/tiff_data/151508/spatial/tissue_hires_image_1.png'
    # moving_path = '/media/huifang/data/fiducial/tiff_data/151509/spatial/tissue_hires_image_vispro.png'
    fixed_path = '/media/huifang/data/sennet/xenium/1812/image_channel/focus0000/channel00_quarter.png'
    moving_path = '/media/huifang/data/sennet/codex/20250314_Yang_SenNet_S4/per_tissue_region-selected/reg001/channel0_half.png'

    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # Get original sizes
    fixed_size = fixed_image.GetSize()
    moving_size = moving_image.GetSize()

    # Compute geometric mean target size
    target_width = int((fixed_size[0] * moving_size[0]) ** 0.5)
    target_height = int((fixed_size[1] * moving_size[1]) ** 0.5)
    target_size = [target_width, target_height]
    print(target_size)
    # target_size = [1024,1024]

    # Resize both images to same target size
    fixed_image = resize_to_target(fixed_image, target_size)
    moving_image = resize_to_target(moving_image, target_size)



    # fixed_image = sitk.Shrink(fixed_image, [4, 4])
    # moving_image = sitk.Shrink(moving_image, [4, 4])
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

    # Resample the moving image using the final transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered_image = resampler.Execute(moving_image)

    # Convert images to numpy arrays for visualization
    fixed_array = sitk.GetArrayViewFromImage(fixed_image)
    moving_array = sitk.GetArrayViewFromImage(moving_image)
    registered_array = sitk.GetArrayViewFromImage(registered_image)


    fixed_norm = normalize(fixed_array)
    moving_norm = normalize(moving_array)
    registered_norm = normalize(registered_array)

    overlay_before = create_overlay(fixed_norm, moving_norm)
    overlay_after = create_overlay(fixed_norm, registered_norm)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(fixed_norm, cmap='gray')
    axes[0].set_title("Fixed Image")

    axes[1].imshow(overlay_before)
    axes[1].set_title("Overlay Before Registration (Magenta/Cyan)")

    axes[2].imshow(overlay_after)
    axes[2].set_title("Overlay After Registration (Magenta/Cyan)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
