#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple


class DNN():
    def __init__(self, features, classes, optimizer, learning_rate):
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

        if optimizer == 'Adam':
            self.prediction = self.add_layer(self.l2,
                                            in_size=68,
                                            out_size=classes)                                    
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.prediction)
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        elif optimizer == 'GradientDescent':
            self.prediction = self.add_layer(self.l2,
                                            in_size=68,
                                            out_size=classes,
                                            activation_function=tf.nn.softmax)
            # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.prediction)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction+1e-20), reduction_indices=[1]))
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
    

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
            X = pairs[:, :features]
            y = pairs[:, features:]

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

    def get_metrics(self, predict_y, real_y, classes=6):
        batch_size = len(predict_y)
        shape = [classes, 1]
        precision = np.zeros(shape)
        recall = np.zeros(shape)
        f1 = np.zeros(shape)
        TP = np.zeros(shape)
        FP = np.zeros(shape)
        FN = np.zeros(shape)

        for i in range(batch_size):
            predicted_class = predict_y[i]
            real_class = real_y[i]
            if predicted_class == real_class:
                TP[predicted_class] += 1
            else:
                FP[predicted_class] += 1
                FN[real_class] += 1
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision*recall) / (precision+recall)
        # Ref: Micro-average vs Macro-average
        # http://sofasofa.io/forum_main_post.php?postid=1001112
        # https://blog.argcv.com/articles/1036.c
        micro_prec = np.asscalar(sum(TP) / (sum(TP)+sum(FP)))
        micro_recall = np.asscalar(sum(TP) / (sum(TP)+sum(FN)))
        micro_f1 = 2*(micro_prec*micro_recall)/(micro_prec+micro_recall)

        macro_prec = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        return (precision, recall, f1,
                micro_prec, micro_recall, micro_f1,
                macro_prec, macro_recall, macro_f1
                )


def train_test_split(raw, random_state=42):
    entries = raw.columns
    want = raw[entries]

    train = want.sample(frac=.8, random_state=random_state)
    # train = want.sample(frac=.8)
    y_train = train[['Activities_Types']]
    y_train = pd.get_dummies(y_train, columns=['Activities_Types'])
    X_train = train.drop(columns=['Activities_Types'])
    X_train = (X_train - X_train.mean()) / X_train.std()
    # X_train = (X_train - X_train.mean())

    test = want.drop(train.index)
    y_test = test[['Activities_Types']]
    y_test = pd.get_dummies(y_test, columns=['Activities_Types'])
    X_test = test.drop(columns=['Activities_Types'])
    X_test = (X_test - X_test.mean()) / X_test.std()
    # X_test = (X_test - X_test.mean())

    return (X_train, X_test, y_train, y_test)


def PCA(X, y, n=2, raw=None):
    if raw != None:
        entries = raw.columns
        want = raw[entries]
        X = want.drop(columns=['Activities_Types'])
        y = want[['Activities_Types']]    
        X = X.values
        y = y.values
    else:
        X = X.values
        y = np.argmax(y.values, axis=1)

    [batch_size, features] = X.shape

    covMat = np.cov(X, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    idx = eigVals.argsort()[::-1]
    W = eigVects[:, idx[:n]]
    low_dim = X*W

    # colors = ['red','green','blue','cyan','purple','yellow']
    colors = ['red','cyan','yellow','green','purple','blue']
    groups = ['dws','ups','sit','std','wlk','jog']
    for i in range(batch_size):
        label = np.asscalar(y[i])

        plt.scatter(low_dim[i, 0],
                    low_dim[i, 1],
                    c=colors[label-1],
                    label=groups[label-1],
                    s=10)
    plt.show()

def tSNE():
    pass



if __name__ == "__main__":
    raw = pd.read_csv('Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(raw)

    PCA(X=X_test, y=y_test, n=2)

    setups = []
    Hyper_parameters = namedtuple('Hyper_parameters', 'optimizer, learning_rate, epochs, batch_size')
    setups.append(Hyper_parameters( optimizer='GradientDescent',
                                    learning_rate=0.5,
                                    epochs=2000,
                                    batch_size=256))
    # setups.append(Hyper_parameters( optimizer='Adam',
    #                                 learning_rate=0.02,
    #                                 epochs=150,
    #                                 batch_size=128))

    # for setup in setups:
    #     dnn = DNN(features=68, classes=6, optimizer=setup.optimizer, learning_rate=setup.learning_rate)
    #     train_loss = dnn.train( X=X_train.values.astype(np.float32),
    #                             y=y_train.values.astype(np.float32),
    #                             epochs=setup.epochs,
    #                             batch_size=setup.batch_size)
    #     print("Train avg cross entropy:{}".format(np.mean(train_loss)))

    #     test_loss = dnn.evaluate(X=X_test.values.astype(np.float32),
    #                             y=y_test.values.astype(np.float32))
    #     print("Test avg cross entropy:{}".format(np.mean(test_loss)))

    #     train_acc = dnn.get_accuracy(dnn.predict(X_train), np.argmax(y_train.values, axis=1)) 
    #     print("Train accuracy:{}".format(train_acc))

    #     predicted_class = dnn.predict(X_test)
    #     real_class = np.argmax(y_test.values, axis=1)
    #     test_acc = dnn.get_accuracy(predicted_class, real_class)
    #     precision, recall, f1, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1 = dnn.get_metrics(predicted_class, real_class)
    #     print("Test accuracy:{}".format(test_acc))
    #     for i in range(6):
    #         print("Test class {} precision:{}".format(i, precision[i]))
    #         print("Test class {} recall:{}".format(i, recall[i]))
    #         print("Test class {} f1-score:{}".format(i, f1[i]))
    #     print("Test micro average precision:{}".format(micro_prec))
    #     print("Test micro average recall:{}".format(micro_recall))
    #     print("Test micro average f1:{}".format(micro_f1))
    #     print("Test macro average precision:{}".format(macro_prec))
    #     print("Test macro average recall:{}".format(macro_recall))
    #     print("Test macro average f1:{}".format(macro_f1))

#%%
