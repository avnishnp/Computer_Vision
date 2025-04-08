import cv2
import glob
from display import Display
from extractor import Frame, denormalize, match_frames, add_ones
import numpy as np
from pointmap import Map, Point

# Camera intrinsics
W, H = 1920 // 2,  1080 // 2
# F = 270
F = 450
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])


Kinv = np.linalg.inv(K)

display = Display(1920, 1080)
mapp = Map()
mapp.create_viewer()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret

def process_frame(img):

    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    # previous frame f2 to the current frame f1.
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    print(f"=------------Rt {Rt}")

    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), 2, (77, 243, 255))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))
        cv2.circle(img, (u2, v2), 2, (204, 77, 255))

    # Use SDL2 display instead of mapp viewer
    display.paint(img)

    mapp.display()
    mapp.display_image(img)


if __name__== "__main__":
    cap = cv2.VideoCapture("videos/car.mp4")

    while True:
        ret, frame = cap.read()
        print("\n#################  [NEW FRAME]  #################\n")
        if not ret:  # Instead of breaking, handle end of video gracefully
            print("Video has ended or failed to read the frame.")
            break
        
        process_frame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()