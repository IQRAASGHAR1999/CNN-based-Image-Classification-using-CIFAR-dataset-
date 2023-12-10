#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Iqra Ashar
#402136
#Assignment 04
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0


# In[2]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
   
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[12]:


model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(32, 32, 3)))  # First Layer
model.add(layers.Conv2D(28, (1, 1), activation='relu', padding='same'))                          # Second Layer
model.add(layers.Conv2D(28, (3, 3), activation='relu', padding='same'))                          # Third Layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(20, (7, 7), activation='relu', padding='same'))                          # Fourth Layer
model.add(layers.Conv2D(28, (7, 7), activation='relu', padding='same'))                          # Fifth Layer
model.add(layers.Conv2D(12, (7, 7), activation='relu', padding='same'))                          # Sixth Layer
model.add(layers.Conv2D(8, (1, 1), activation='relu', padding='same'))   
model.add(layers.MaxPooling2D((2, 2)))

model.summary()


# In[13]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# In[11]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[ ]:




