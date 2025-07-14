from PIL import Image
from matplotlib import pyplot as plt


def crop_image_with_absolute_pixels(image_path, top_left, bottom_right, output_path):
    """
    Crop an image directly using absolute pixel coordinates.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    top_left : tuple(int, int)
        (x0, y0) pixel coordinates for the top-left corner.
    bottom_right : tuple(int, int)
        (x1, y1) pixel coordinates for the bottom-right corner.
    output_path : str
        File path to save the cropped image.
    """

    # 1. Open image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    # 2. Define crop box = (left, upper, right, lower)
    box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    # 3. Crop and save
    cropped_img = img.crop(box)
    plt.imshow(cropped_img)
    plt.show()

    cropped_img.save(output_path)
    print(f"Cropped image saved to: {output_path}")


# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    input_image = "/media/huifang/data/registration/humanpilot/151509/spatial/tissue_hires_image_image_0.png"
    # Suppose top_left = (100, 100), bottom_right = (400, 400)
    crop_image_with_absolute_pixels(
        image_path=input_image,
        top_left=(118, 373),
        bottom_right=(1765, 2015),
        output_path="/media/huifang/data/registration/humanpilot/151509/spatial/tissue_hires_image_cropped.png"
    )
