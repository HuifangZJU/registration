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


from sklearn.neighbors import NearestNeighbors
def spotwise_topk_accuracy(coords_A, coords_B, labels_A, labels_B, dist_thresh=5.0):
    offset = 5


    # normalize pts1 to [0, 10]
    min_vals = coords_A.min(axis=0)
    max_vals = coords_B.max(axis=0)
    scale = max_vals - min_vals

    coords_A = (coords_A - min_vals) / scale * 100 + offset
    coords_B = (coords_B - min_vals) / scale * 100 + offset

    nn = NearestNeighbors(radius=dist_thresh, algorithm="auto").fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)

    nB = coords_B.shape[0]
    correct_majority, correct_any, count_valid = 0, 0, 0

    for i in range(nB):
        idxs = neighbors[i]
        if len(idxs) == 0:
            # no neighbors in threshold → skip this spot
            continue
        neigh_labels = labels_A[idxs]
        true_label = labels_B[i]

        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        maj_label = vals[np.argmax(counts)]
        if maj_label == true_label:
            correct_majority += 1

        # relaxed: if true label appears among neighbors
        if true_label in neigh_labels:
            correct_any += 1

        count_valid += 1

    acc_majority = correct_majority / count_valid if count_valid > 0 else np.nan
    return acc_majority



import os
root = "/media/huifang/data/registration/SCC/huifang/ablation/image"
for patient in [2,5,9,10]:
    print(patient)
    for ablation_level in [1,2,3,4,5]:
        print(ablation_level)
        results=[]
        for pair in [0,1]:
            path = os.path.join(root,str(ablation_level))
            pair0 = path+ f"/patient_{patient}_{pair}"
            pair1 = path+ f"/patient_{patient}_{pair+1}"
            fixed_image_path = pair0 + "_image_512.png"
            moving_image_path = pair1 + "_image_512.png"
            displacement_field,fixed_image,moving_image, warped_image = get_simpleITK_transformation(fixed_image_path,moving_image_path)
            print('calculation is done, perform transformations..')

            fixed_val = np.load(pair0 + '_validation.npz')
            fixed_pts = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(pair1 + '_validation.npz')
            moving_pts = moving_val["coord"]
            moving_label = moving_val["label"]

            # warped_pts = moving_pts
            warped_pts = transform_uv_with_displacement(moving_pts, displacement_field)

            topk_acc = spotwise_topk_accuracy(fixed_pts, warped_pts, fixed_label, moving_label)
            results.append(topk_acc)

        results.append(np.mean(np.asarray(results)))
        print(results)

