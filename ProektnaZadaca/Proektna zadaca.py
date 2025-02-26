import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def create_automated_mask(image, lower_thresh=200, upper_thresh=255):
    """Creates a mask to target damaged white areas using thresholding and contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Threshold the image to get binary mask of damaged areas
    _, binary_mask = cv2.threshold(gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY)

    # Find contours and fill them in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask

def load_external_mask(mask_path, image_shape):
    """Loads an external mask from a file if available, and adjusts its size."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")
    # Resize the mask to match the image size
    mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]))
    return mask_resized

def apply_inpainting(image, mask, method):
    """Applies inpainting to the image using the specified method."""
    inpaint_radius = 3  # You can experiment with the radius for better results
    if method == 'telea':
        return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == 'ns':
        return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_NS)
    else:
        raise ValueError("Invalid inpainting method. Choose 'telea' or 'ns'.")

def save_and_display_images(original, mask, inpainted_images):
    """Displays the original image, mask, and the inpainted results."""
    cv2.imshow('Original Image', original)
    cv2.imshow('Mask', mask)

    # Display results of both inpainting methods
    for method_name, img in inpainted_images.items():
        cv2.imshow(f'Inpainted Image - {method_name}', img)
        # Optionally, save the images
        output_path = f'/mnt/data/inpainted_image_{method_name}.jpeg'
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the original image
    image_path = 'slika1.jpeg'
    image = load_image(image_path)

    # Create an automated mask
    mask = create_automated_mask(image)

    # Apply both inpainting algorithms: 'telea' and 'ns' (Navier-Stokes)
    inpainted_telea = apply_inpainting(image, mask, 'telea')
    inpainted_ns = apply_inpainting(image, mask, 'ns')

    # Save and display the results
    inpainted_images = {
        'Telea': inpainted_telea,
        'Navier-Stokes': inpainted_ns
    }
    save_and_display_images(image, mask, inpainted_images)

if __name__ == "__main__":
    main()