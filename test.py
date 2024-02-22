import tensorflow
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

#keras.Sequential
#this approach is "theoretical", not applied in real project
seq_model = keras.Sequential(
    [
        Input(shape=(28,28,1)), #img: 28x28 pixel from mnist, 1 channel because of grayscaled
        #Conv2D need 4D tensor, (28,28,1) is 4D since 1 dim is hidden
        #parameter to train model, decide model's accuracy
        Conv2D(32, (3,3), activation='relu'), #32 filter, each one size 3x3
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'), #filter can be change to whatever
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(), 
        Dense(64, activation='relu'), #units can be chosen differently
        #output layer:
        Dense(10, activation='softmax'), #mnist dataset classifies 0-9 handwritting -> 10 units
    ]
)

#functional approach: function that returns a model
#this approach is good and useful for real project
def functional_model(): #flexible
    #method's parameter could be img size, channel, filter...
    my_input = Input(shape=(28,28,1)) #img: 28x28 pixel from mnist, 1 channel because of grayscaled
    #Conv2D need 4D tensor, (28,28,1) is 4D since 1 dim is hidden
    #parameter to train model, decide model's accuracy
    x = Conv2D(32, (3,3), activation='relu')(my_input) #32 filter, each one size 3x3
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x) #filter can be change to whatever
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x) #units can be chosen differently
    #output layer:
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs = my_input, outputs = x)

    return model


#keras.Model: inherit from this class
class MyCustomModel(keras.Model):

    def __init__(self)->None:
        super().__init__()

        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def display_some_examples(examples, labels):
    
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    
    plt.show()

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", x_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False: #pass this code for now
        display_some_examples(x_train, y_train)
    
    #normalize data: RGB -> 255
    x_train = x_train.astype('float32') / 255 #x, y are unsigned integers
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1) #add dimension at the end
    x_test = np.expand_dims(x_test, axis=-1) #can change axis=3 here and above

    #convert to one-hot label to use loss='categorical_crossentropy'
    #if not, have to use loss='sparse_categorical_crossentropy'
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if False:
        model = functional_model()
    
    model = MyCustomModel()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    #optimizer: algorithms, like gradient descent...
    #loss: loss function
    #metrics: judge the model's performance

    #model training:
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    #batch_size: number of images are seen each time, change for experimentation 
    #epoch: times of traversing through entire dataset
    #validation_split: split dataset into train set and validate set

    #evaluation on test set:
    model.evaluate(x_test, y_test, batch_size=64)