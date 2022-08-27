import os
from unicodedata import name
from tensorflow.keras.layers import Input,LSTM, Dense, InputLayer, Flatten, dot
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from preprocessing import preprocess




def train(signs_X1, signs_X2, signs_y):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    # Design model
    
    input_layer = Input((30,1662))
    layer1 = LSTM(128, return_sequences=True, activation='tanh')(input_layer)
    layer2 = LSTM(256, return_sequences=True, activation='tanh')(layer1)
    layer3 = LSTM(128, return_sequences=False, activation='tanh')(layer2)
    layer4 = Dense(128, activation='relu')(layer3)
    layer5 = Dense(64, activation='relu')(layer4)
    layer5 = Flatten()(layer5)

    embeddings = Dense(32, activation=None)(layer5)
    norm_embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        
    # Create model
    model = Model(inputs=input_layer, outputs=norm_embeddings)


    #Create siamese model 
    input1 = Input((30,1662))
    input2 = Input((30,1662))

    # Create left and right twin models
    left_model = model(input1)
    right_model = model(input2)


    # Dot product layer
    dot_product = dot([left_model, right_model], axes=1, normalize=False)

    siamese_model = Model(inputs=[input1, input2], outputs=dot_product)

    # Compile model    
    siamese_model.compile(optimizer='adam', loss= 'mse', metrics=['categorical_accuracy'])
    

    # Fit model
    siamese_model.fit([signs_X1, signs_X2], signs_y, epochs=100, batch_size=5, shuffle=True, verbose=True)


    model.save(os.getcwd()+"/sign_language_encoder.h5")
    siamese_model.save(os.getcwd()+"/siamese_SL_model.h5")

def main():
    signs_X1, signs_X2, signs_y = preprocess()
    train(signs_X1, signs_X2, signs_y)

if __name__ == '__main__':
    main()