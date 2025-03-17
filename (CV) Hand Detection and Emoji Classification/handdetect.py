import cv2
import numpy as np

def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Video not established. Exiting...")
        exit()

    while True:
        success, frame = video.read()
        if not success:
            print("Stream end")
            break

        # Flip the frame for a better user experience (mirror the image)
        frame = cv2.flip(frame, -1)

        # Define the region of interest (ROI) - adjust this box size and position as needed
        roi = frame[50:500, 50:500]  # Example ROI, adjust as needed

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Apply adaptive thresholding to segment the hand from the background
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 15, 5)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty black mask for the hand (same size as the ROI)
        hand_mask = np.zeros_like(thresholded)  # Create a binary mask based on thresholded image

        # Fill the hand region with white on the mask
        cv2.drawContours(hand_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Create a white image (same size as the ROI) to place the filled hand
        filled_hand_image = np.ones_like(roi) * 255  # White background image

        # Use the mask on the white image to display the filled hand
        filled_hand = cv2.bitwise_and(filled_hand_image, filled_hand_image, mask=hand_mask)

        # Show the filled hand on a black background (in the original frame)
        frame[50:500, 50:500] = filled_hand

        # Display the result
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
