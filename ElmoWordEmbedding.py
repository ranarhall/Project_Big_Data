import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

def ElmoEmbedding(tokens_input):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())

    batch = 1000

    vectors = []

    for i in range(int(len(tokens_input)/batch)+1) :
        embeddings = elmo(tokens_input[i*batch:(i+1)*batch])
        temp = sess.run(embeddings)
        vectors.extend(temp)
    sess.close()

    return vectors

def GetTokensEmbedding(words, embeddings, sentences):
    return [[embeddings[words.index(word)] for word in tokens] for tokens in sentences]
