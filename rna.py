''' 
Nome: Gabriel Gomes Cruz da Rocha       Matricula: 201407708
Rede Neural  de Reconhecimento de Imagens
'''

'''
Importa os modulos necessarios para criar a rede neural de reconhecimento de 
imagens a partir do Keras nativo do Tensorflow.
'''
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

epochs = 20
batch_size = 32

'''Selecionamos nosso conjunto de Treino e Teste'''

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'cell_images/training_set',
        target_size = (28, 28),
        batch_size = batch_size,
        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
        'cell_images/test_set',
        target_size = (28, 28),
        batch_size = batch_size,
        class_mode = 'binary')

''' Criamos a Rede neural como sequencial e com 3 camadas alem da camada de
    Flatten'''
    
classifier = Sequential([
        Flatten(input_shape=(28,28,3)),
        Dense(500, activation = 'relu'),
        Dense(500, activation = 'tanh'),
        Dense(1, activation = 'sigmoid')])

'''Compilamos o modelo criado'''

classifier.compile(
        optimizer = 'adam', 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])

steps_per_epoch = len(training_set)/batch_size
validation_steps = len(test_set)/batch_size

H = classifier.fit_generator(
        training_set,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = test_set,
        validation_steps = validation_steps)

classifier.save_weights('classifier.h5')

score = classifier.evaluate_generator(test_set)

print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.figure()
plt.plot(np.arange(0,epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,epochs), H.history["val_acc"], label="val_acc")
plt.title("Acuracia")
plt.xlabel("Épocas #")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.title("Perda")
plt.xlabel("Épocas #")
plt.ylabel("Accuracy")
plt.legend()
plt.show()