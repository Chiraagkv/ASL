class Slicer ():
    def __init__(self, image):
        self.image=image
    def process_complete (self):
        list_full=[]
        img=tf.io.read_file(self.image)
        img=tf.image.decode_image(img)
        img=tf.image.resize(img, size=(224, 224))
        img=img/255.
        lis=self.image_slicer(image=img)
        for i in range (len(lis)):
            ans, ans2=self.image_2(lis[i])
            list_full.append(ans)
            list_full.append(ans2)
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
      
      
      
      """
      pass it like this
      image=myimage.jpg
      a=Slicer(image)
      list_of_img=a.process_complete()
      
      for i in range (len(list_of_img)):
        plt.imshow(list_of_img[i])
      """
