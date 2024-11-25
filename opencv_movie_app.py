import cv2
import numpy as np

image_path = 'img.png'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    print(f"Contour {i + 1}:")
    print(contour)
    print()

clips = ["video.mp4", "video2.mp4", "video3.mp4", "video.mp4", "video5.mp4", "video5.mp4", "video7.mp4"]
video_clips = [cv2.VideoCapture(clip) for clip in clips]

fps = video_clips[0].get(cv2.CAP_PROP_FPS)

frame_count = int(10 * fps)

assert len(contours) >= len(video_clips), "The number of videos should not exceed the number of contours."

frame_width, frame_height = image.shape[1], image.shape[0]
out = cv2.VideoWriter('composition_fixed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for frame_idx in range(frame_count):
    output_frame = image.copy()

    for clip_idx, (video_clip, contour) in enumerate(zip(video_clips, contours)):
        ret, frame = video_clip.read()

        if not ret:
            print(f"No more frames to read from video {clip_idx + 1}")
            continue

        x, y, w, h = cv2.boundingRect(contour)

        resized_frame = cv2.resize(frame, (w, h))

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mask = mask[y:y + h, x:x + w]
        mask_inv = cv2.bitwise_not(mask)
        roi = output_frame[y:y + h, x:x + w]

        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

        dst = cv2.add(bg, fg)

        output_frame[y:y + h, x:x + w] = dst

    out.write(output_frame)

    cv2.imshow('Output Frame', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


for video_clip in video_clips:
    video_clip.release()
out.release()
cv2.destroyAllWindows()
