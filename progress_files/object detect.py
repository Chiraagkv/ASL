import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas
import sys
import cv2
import time
import matplotlib.pyplot as plt
import tempfile
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import pickle


module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
model = hub.load(module_handle)
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def load_image(path, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    pil_image = Image.open(path)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    
    print("Image downloaded to %s." % filename)
    
    if display==True:
        display_image(pil_image)
    else:
      pass

    
    return filename

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=(),
                               draw_text_for_asl=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)


    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if draw_text_for_asl==True:
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)
            text_bottom -= text_height - 2 * margin
    else:
        pass


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    colors = list(ImageColor.colormap.values())
    f=open('predictions.txt', 'w')

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            if class_names[i].decode('ascii')=='Human hand':
              display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
              color = colors[hash(class_names[i]) % len(colors)]
              image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
              draw_bounding_box_on_image(image_pil,
                                         ymin,
                                         xmin,
                                         ymax,
                                         xmax,
                                         color,
                                         font,
                                         display_str_list=[display_str])
              main_str=f"{ymin}-{xmin}-{ymax}-{xmax}"
              f.write(main_str)
              np.copyto(image, np.array(image_pil))
              return image


            else:
              pass
def run_detector(frames):
  img = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
  converted_img  = tf.image.convert_image_dtype(frames, tf.float32)[tf.newaxis, ...]
  result = model.signatures['default'](converted_img)
  result = {key:value.numpy() for key,value in result.items()}
  for m in range (len(result['detection_class_entities'])):
    if b"Human hand"==result['detection_class_entities'][m]:
      image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])
      print("Hand Present")
      #display_image(image_with_boxes)
      break
    else:
      pass

video=cv2.VideoCapture(0)
while True:
    _, frame=video.read()
    run_detector(frames=frame)
    if cv2.waitkey(0) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyallwindows()
