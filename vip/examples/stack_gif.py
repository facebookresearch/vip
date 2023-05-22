from PIL import Image, ImageSequence

# load GIFs
gif1 = Image.open('/home/exx/Projects/vip/vip/assets/fold_towel_vip.gif')
gif2 = Image.open('/home/exx/Projects/vip/vip/assets/kitchen_sdoor_open-v3_vip.gif')
gif3 = Image.open('/home/exx/Projects/vip/vip/assets/task_hammer_vip.gif')

frames = []

for frame1, frame2, frame3 in zip(ImageSequence.Iterator(gif1),
                                   ImageSequence.Iterator(gif2),
                                   ImageSequence.Iterator(gif3)):
    # stack frames
    new_frame = Image.new('RGBA', (frame1.width, frame1.height*3))
    new_frame.paste(frame1, (0, 0))
    new_frame.paste(frame2, (0, frame1.height))
    new_frame.paste(frame3, (0, frame1.height*2))
    frames.append(new_frame)

for i in range(30):
    frames.append(new_frame)
# save result
frames[0].save('stacked_gif.gif', save_all=True, append_images=frames[1:], loop=0, duration=50)