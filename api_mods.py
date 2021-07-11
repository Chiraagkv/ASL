def model_make (data_path, checkpoints=False, model_type='api', show_loss_curves=False, api_dense_layer_activate):
  def show_crv (boolean, hist):
    if boolean==True:
      lrs=1e-4*10**(np.arange(0, 50)/200)
      plt.semilogx(lrs, hist.history['loss'])
    else:
      pass 
  if model_type.lower()=="cnn":
    data=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                     samplewise_std_normalization=True,
                                                     height_shift_range=0.2,
                                                     width_shift_range=0.2,              
                                                     rotation_range=0.2,
                                                     zca_epsilon=1e-6,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     cval=0.0)
    data=data.flow_from_directory(directory=data_path,
                                  target_size=(224, 224),
                                  class_mode='categorical',
                                  batch_size=19,
                                  shuffle=False,
                                  interpolation='bilinear')
    model=tf.keras.Sequential([
                             tf.keras.layers.Conv2D(filters=128,
                                                    kernel_size=2,
                                                    padding='same',
                                                    activation='relu'),
                             tf.keras.layers.Conv2D(128, 3, activation='relu'),
                             tf.keras.layers.Conv2D(128, 2, activation='relu'),

                             tf.keras.layers.Conv2D(64, 3, activation='relu'),
                             tf.keras.layers.Conv2D(64, 3, activation='relu'),
                             tf.keras.layers.Conv2D(64, 3, activation='relu'),

                             tf.keras.layers.Conv2D(32, 2, activation='relu'),
                             tf.keras.layers.MaxPool2D(pool_size=2,
                                                       padding='same',
                                                       data_format=None),
                             tf.keras.layers.Conv2D(32, 2, activation='relu'),
                             tf.keras.layers.Flatten(),

                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),

                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(64, activation='relu'),

                             tf.keras.layers.Dense(32, activation='relu'),
                             tf.keras.layers.Dense(32, activation='relu'),
                             tf.keras.layers.Dense(32, activation='relu'),

                             tf.keras.layers.Dense(16, activation='relu'),
                             tf.keras.layers.Dense(16, activation='relu'),
                             tf.keras.layers.Dense(16, activation='relu'),


                             tf.keras.layers.Dense(len(os.listdir(data_path)), activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics='accuracy')
    if checkpoints==False:
      history=model.fit(data, 
              epochs=50,
              steps_per_epoch=len(data),
              callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200))
                        tf.keras.callbacks.ModelChecpoint(log_dir='checkpoints.ckpt',
                                                         monitor='accuracy',
                                                         save_weight_only=True,
                                                         verbose=1)]
                       )
    else:
      inpus=str(input("Enter the filename (**Full path**): "))
      try:
        model=tf.keras.load_model(inpus)
      except Exception as e:
        print ("Enter the full path")
    show_crv(boolean=show_loss_curves, hist=history)
    return model

  if model_type=='api':

    data=tf.keras.preprocessing.image_dataset_from_directory(
      directory=data_path,
      label_mode='categorical',
      color_mode='rgb',
      interpolation='nearest',
      image_size=(224, 224),
      shuffle=False,
      smart_resize=True,
      batch_size=19,
      follow_links=True
    )

    base_model=tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(224, 224, 3))

    inputs=tf.keras.Input(shape=(224, 224, 3))

    mods=base_model(inputs)

    layer_1=tf.keras.layers.AveragePooling2D()(mods)
    layer_2=tf.keras.layers.GlobalAveragePooling2D()(layer_1)
    layer_3=tf.keras.layers.GlobalMaxPool2D()(tf.expand_dims(tf.expand_dims(layer_2, axis=0), axis=0))

    outputs=tf.keras.layers.Dense(128, activation='relu')(layer_3)
    outputs=tf.keras.layers.Dense(128, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(128, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(64, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(64, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(64, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(32, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(32, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(32, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(16, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(16, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(16, activation='relu')(outputs)

    output_real=tf.keras.layers.Dense(len(os.listdir(data_path)), activation='softmax')(outputs)

    model=tf.keras.Model(inputs, output_real)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics='accuracy')
    
    if checkpoints==False:
      history=model.fit(data, 
              epochs=50,
              steps_per_epoch=len(data),
              callbacks=tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200)))
    if callbacks==True:
      for roots, names, filenames in os.walk(os.getcwd()):
        for m in filenames:
          lists=filenames.split('.')
          if lists[0]=='checkpoint':
            history=model.fit(data,
                              epochs=1)
            model.load_weights('checkpoints.ckpt')
            break
          else:
            continue 

    show_crv(boolean=show_loss_curves)
    return model

  
