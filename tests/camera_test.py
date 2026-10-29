import cv2

from app.camera.camera import Camera


def main():
    """Open the camera, display the live feed, and close on user input."""

    with Camera() as camera:
        while True:
            _, frame = camera.read()

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()