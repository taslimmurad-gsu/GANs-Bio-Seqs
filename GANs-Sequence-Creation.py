import numpy as np
import tensorflow as tf

def get_gans(input_sequences, total_gan_sequences, filename):
    def generator():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(len(input_sequences[0]),)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(len(input_sequences[0]), activation='softmax'))
        return model
    def discriminator():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(len(input_sequences[0]),)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    generator_model = generator()
    discriminator_model = discriminator()
    discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    def gan(generator, discriminator):
        model = tf.keras.Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    print(generator_model.summary())
    print(discriminator_model.summary())
    gan_model = gan(generator_model, discriminator_model)
    total_gan_sequences = total_gan_sequences
    each_seq_length = len(input_sequences[0])
    total_iterations = 100
    batch_size = 32
    for i in range(total_iterations):
        if (i % 10000 == 0):
            print(i)
        idx = np.random.randint(0, input_sequences.shape[0], size=batch_size)
        real_seq = input_sequences[idx]
        noise = np.random.rand(total_gan_sequences, each_seq_length)
        generated_seq = generator_model.predict(noise)
    print('generated_seq shape', generated_seq.shape)
    np.save(filename, np.array(generated_seq))
    print('done')

sequences = np.load("path to input sequences .npy file")
attribute_data = np.load("path to target labels .npy file")
print(np.unique(attribute_data, return_counts=True))

### create list for each target label
seqs_gans_target1 = []
seqs_gans_target2 = []
seqs_gans_target3 = []

### get the sequences corresponding to each target label in their respective lists.
for i in range(len(attribute_data)):
    if(attribute_data[i]=='bat'):
        seqs_gans_target1.append(sequences[i])
    if (attribute_data[i] == 'bird'):
        seqs_gans_target2.append(sequences[i])
    if (attribute_data[i] == 'bovine'):
        seqs_gans_target3.append(sequences[i])

print('len seqs_gans_target1', len(seqs_gans_target1))
print('len seqs_gans_target2', len(seqs_gans_target2))
print('len seqs_gans_target3', len(seqs_gans_target3))

### create the GANs-based data for the respective target label and save it in the given file
input_sequences = np.array(seqs_gans_target1)
print('input_sequences shape', input_sequences.shape)
total_gan_sequences = len(seqs_gans_target1)                ### no of GANs sequences to create
filename = 'path of .npy file to save the GANs-based created data'
get_gans(input_sequences, total_gan_sequences, filename)
print('seqs_gans_target1 done')

### create the GANs-based data for the respective target label and save it in the given file
input_sequences = np.array(seqs_gans_target2)
print('input_sequences shape', input_sequences.shape)
total_gan_sequences = len(seqs_gans_target2)                ### no of GANs sequences to create
filename = '/alina-data2/taslim/GANs_CHIL/Host/Host_GANS_bird_PWM2Vec_full.npy'
get_gans(input_sequences, total_gan_sequences, filename)
print('seqs_gans_target2 done')

### create the GANs-based data for the respective target label and save it in the given file
input_sequences = np.array(seqs_gans_target3)
print('input_sequences shape', input_sequences.shape)
total_gan_sequences = len(seqs_gans_target3)                ### no of GANs sequences to create
filename = '/alina-data2/taslim/GANs_CHIL/Host/Host_GANS_bovine_PWM2Vec_full.npy'
get_gans(input_sequences, total_gan_sequences, filename)
print('seqs_gans_target3 done')