import tensorflow as tf
class Slicer ():
    def __init__(self, image):
        self.image=image
        
    def image_2 (self,image):
      img_1t=tf.image.rot90(image=image, k=1)

      first=tf.keras.layers.Cropping1D(cropping=(150,1))(img_1t.numpy())

      second=tf.image.flip_left_right(img_1t.numpy())

      second=tf.keras.layers.Cropping1D(cropping=(150,1))(second.numpy())

      first=tf.image.rot90(first, k=3)
      second=tf.image.rot90(second, k=3)
      return first, second
    def process_complete (self):
        list_full=[]
        img=tf.io.read_file(self.image)
        try:
          img=tf.image.decode_image(img)
        except Exception as e:
          img=tf.image.decode_jpeg(img)
        img=tf.image.resize(img, size=(224, 224))
        print (img.shape)
        img=img/255.
        lis=self.image_slicer(image=img)
        for i in range (len(lis)):
            ans, ans2=self.image_2(lis[i])
            list_full.append(tf.image.resize(ans, size=(224, 224)))
            list_full.append(tf.image.resize(ans2, size=(224, 224)))
        return list_full

    def block_1 (self,image):
        image_b3=tf.image.flip_left_right(image)
        image_b3=tf.keras.layers.Cropping1D(cropping=(150, 1))(image_b3)
        image_b3=tf.image.flip_left_right(image_b3)
        return image_b3


    def block_2 (self,image):
        image_flipped_centre=tf.image.flip_left_right(image)

        image_flipped_centre=tf.keras.layers.Cropping1D(cropping=(90, 1))(image_flipped_centre)

        image_flipped_2=tf.image.flip_left_right(image_flipped_centre)
        image_flipped_2=tf.keras.layers.Cropping1D(cropping=(50, 1))(image_flipped_2)
        return image_flipped_2


    def block_3(self,image):
        #image_b1=tf.image.flip_left_right(image)
        image_b1=tf.keras.layers.Cropping1D(cropping=(150, 1))(image)
        return image_b1
    


    def image_slicer (self,image):

        image_full=tf.image.rot90(image=image)
        img1=self.block_1(image=image_full.numpy())

        img2=self.block_2(image=image_full.numpy())

        img3=self.block_3(image=image_full.numpy())
        all=[img1, img2, img3]
        list=[]

        for i in range (len(all)):
          rost=tf.image.rot90(all[i], k=3)
          last=self.block_3(image=tf.image.resize(rost.numpy(), size=(224, 224)))
          cent=self.block_2(image=tf.image.resize(rost.numpy(), size=(224, 224)))
          fir=self.block_1(image=tf.image.resize(rost.numpy(), size=(224, 224)))
          list.append(last)

          list.append(cent)

          list.append(fir)

        return list
      
      
      

list_full=[]

filename=filename
classss=os.listdir(filename)
for m in range (len(classss)):
  main_fold=filename+"/"+ classss[m]
  print ("1")
  for root, dirs, filenames in os.walk(main_fold):
    print ('2')
    print (f"length {len(filenames)}")
    print (filenames)
    for l in range (len(filenames)):
      first_image=filename+'/'+classss[m]+'/'+filenames[0]
      print (first_image)
      i=1
      for i in range (len(filenames)):
        print (i)
        print (filenames[0])
        a=Slicer(filename+'/'+classss[m]+'/'+filenames[i])
        b=Slicer(first_image)
        lis1=a.process_complete()
        lis2=b.process_complete()

        print (len(lis1))
        for k in range (18):
          if (np.array(lis1[k])==np.array(lis2[k])).all():
            os.remove(filename+'/'+classss[m]+'/'+filenames)
            continue
          if (np.array(lis1[k])!=np.array(lis2[k])).all():
            break
        if k==17:
          print ('3')
          print (f" shape {lis2[0].shape}")
          list_full.append(lis2)
          print (f"removed {filename+'/'+classss[m]+'/'+filenames[l]}")
        if k!=17:
          (f"cannot remove {filename+'/'+classss[m]+'/'+filenames[l]}")
          print ('5')
          continue
        print (filenames)
