from pdf2image import convert_from_path

# Set your PDF path
pdf_path = "/home/huifang/workspace/grant/k99/resubmission/figures/roadmap.pdf"

# Convert with high resolution (300+ DPI)
images = convert_from_path(pdf_path, dpi=1200)  # You can increase dpi for better quality

# Save each page as PNG
for i, image in enumerate(images):
    image.save(f"/home/huifang/workspace/grant/k99/resubmission/figures/roadmap_{i+1}.png", "PNG")