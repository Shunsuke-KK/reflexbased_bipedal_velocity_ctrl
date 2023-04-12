import sys
import cv2
import os

# encoder (for mp4)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# output file name, encoder , fps, size (fit to image size)
num = 0
while True:
    filename = f'minmax_vel{num}.mp4'
    if os.path.exists(os.path.join(os.getcwd(),filename)):
        num += 1
    else:
        break
video = cv2.VideoWriter(os.path.join(os.getcwd(),filename), fourcc, 50, (1920,1080))
print('video path -->', os.path.join(os.getcwd(),filename))


if not video.isOpened():
    print('cannot open')
    sys.exit()

pic_num = 3000
pic_file = '0'
pic_path = os.path.join(os.getcwd(),pic_file)
for i in range(0, pic_num):
    img = cv2.imread(os.path.join(pic_path,f'{i}.png'))
    if img is None:
        print('cannot read')
        break

    # add
    video.write(img)
print('generating videos....')
video.release()
