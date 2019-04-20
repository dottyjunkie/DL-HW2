#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class DNN():
    def __init__(self, features, classes):
        self.xs = tf.placeholder(tf.float32, [None, features])
        self.ys = tf.placeholder(tf.float32, [None, classes])

        self.l1 = self.add_layer(self.xs,
                                in_size=features,
                                out_size=68,
                                activation_function=tf.nn.relu)
        
        self.l2 = self.add_layer(self.l1,
                                in_size=68,
                                out_size=68,
                                activation_function=tf.nn.relu)

        self.prediction = self.add_layer(self.l2,
                                        in_size=68,
                                        out_size=classes)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.prediction)
    
        # self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
        self.train_step = tf.train.AdamOptimizer(0.02).minimize(self.loss)

        self.init = init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        
        return outputs

    def train(self, X, y, epochs=1000, batch_size=128):
        total_size, features = X.shape

        for i in range(epochs):
            pairs = np.column_stack((X, y))
            np.random.shuffle(pairs)
            X = pairs[:, :68]
            y = pairs[:, 68:]

            for b in range(int(total_size/batch_size)):
                # print("Batch {}".format(b))
                X_batch = X[b*batch_size:(b+1)*batch_size, :]
                y_batch = y[b*batch_size:(b+1)*batch_size, :]
                self.sess.run(self.train_step, feed_dict={self.xs:X_batch, self.ys:y_batch})

            # if i%10 == 0:        
            #     loss = self.sess.run(self.loss, feed_dict={self.xs:X, self.ys:y})
            #     print("Epoch {} avg cross entropy:{}".format(i, np.mean(loss) ))

        loss = self.sess.run(self.loss, feed_dict={self.xs:X, self.ys:y})
        return loss
        

    def evaluate(self, X, y):
        loss = self.sess.run(self.loss, feed_dict={self.xs:X, self.ys:y})
        return loss

    def predict(self, X):
        predicted_value = self.sess.run(self.prediction, feed_dict={self.xs: X})
        y = np.argmax(predicted_value, axis=1)
        return y

    def get_accuracy(self, predicted_y, real_y):
        batch_size = len(predicted_y)
        errors = 0.0
        for i in range(batch_size):
            if predicted_y[i] != real_y[i]:
                errors += 1
        return 1 - errors/batch_size

    def 





def train_test_split(raw, random_state=42):
    entries = raw.columns
    want = raw[entries]

    # train = want.sample(frac=.8, random_state=random_state)
    train = want.sample(frac=.8)
    y_train = train[['Activities_Types']]
    y_train = pd.get_dummies(y_train, columns=['Activities_Types'])
    X_train = train.drop(columns=['Activities_Types'])
    X_train = (X_train - X_train.mean()) / X_train.std()

    test = want.drop(train.index)
    y_test = test[['Activities_Types']]
    y_test = pd.get_dummies(y_test, columns=['Activities_Types'])
    X_test = test.drop(columns=['Activities_Types'])
    X_test = (X_test - X_test.mean()) / X_test.std()

    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    raw = pd.read_csv('Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(raw)

    dnn = DNN(features=68, classes=6)
    train_loss = dnn.train( X=X_train.values.astype(np.float32),
                            y=y_train.values.astype(np.float32),
                            epochs=100)
    print("Train avg cross entropy:{}".format(np.mean(train_loss)))

    test_loss = dnn.evaluate(X=X_test.values.astype(np.float32),
                            y=y_test.values.astype(np.float32))
    print("Test avg cross entropy:{}".format(np.mean(test_loss)))

    train_acc = dnn.get_accuracy(dnn.predict(X_train), np.argmax(y_train.values, axis=1)) 
    print("Train accuracy:{}".format(train_acc))
    
    test_acc = dnn.get_accuracy(dnn.predict(X_test), np.argmax(y_test.values, axis=1))
    print("Test accuracy:{}".format(test_acc))