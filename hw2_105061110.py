#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DNN():
    def __init__(self, features, classes, optimizer, learning_rate):
        self.xs = tf.placeholder(tf.float32, [None, features])
        self.ys = tf.placeholder(tf.float32, [None, classes])
       

        if optimizer == 'Adam':
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
                                            out_size=classes
                                            # ,activation_function=tf.nn.softmax
                                            )                                    
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.prediction)
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction+1e-30), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        elif optimizer == 'GradientDescent':
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
                                            out_size=classes
                                            ,activation_function=tf.nn.softmax
                                            )     
            # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ys, logits=self.prediction)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction+1e-30), reduction_indices=[1]))
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
    

        self.init = init = tf.global_variables_initializer()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
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

    def train(self, X, y, X_validate, y_validate, epochs=1000, batch_size=128):
        total_size, features = X.shape
        loss = np.zeros((epochs, 1))
        loss_validate = np.zeros((epochs, 1))
        acc = np.zeros((epochs, 1))
        acc_validate = np.zeros((epochs, 1))
        X_whole = X.copy()
        y_whole = y.copy()
        y_train_class = np.argmax(y_whole, axis=1)
        y_test_class = np.argmax(y_validate, axis=1)

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

            loss[i] = np.mean(self.sess.run(self.loss, feed_dict={self.xs:X_whole, self.ys:y_whole}))
            loss_validate[i] = np.mean(self.sess.run(self.loss, feed_dict={self.xs:X_validate, self.ys:y_validate}))
            acc[i] = self.get_accuracy(predicted_y=self.predict(X_whole), real_y=y_train_class)
            acc_validate[i] = self.get_accuracy(predicted_y=self.predict(X_validate), real_y=y_test_class)
        return loss, loss_validate, acc, acc_validate
        

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

    test = want.drop(train.index)
    y_test = test[['Activities_Types']]
    y_test = pd.get_dummies(y_test, columns=['Activities_Types'])
    X_test = test.drop(columns=['Activities_Types'])
    X_test = (X_test - X_test.mean()) / X_test.std()

    return (X_train, X_test, y_train, y_test)


def pca(X, n=2, raw=None):
    if raw is not None:
        entries = raw.columns
        want = raw[entries]
        X = want.drop(columns=['Activities_Types'])
        y = want[['Activities_Types']]    
        X = X.values
    else:
        X = X.values

    covMat = np.cov(X, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    idx = eigVals.argsort()[::-1]
    W = eigVects[:, idx[:n]]
    low_dim = X*W

    return low_dim


def plot2d(X_y_pairs, name=None):
    X = X_y_pairs[0]
    y = X_y_pairs[1]
    [batch_size, features] = X.shape
    colors = ['red','cyan','yellow','green','purple','blue']
    labels = ['dws','ups','sit','std','wlk','jog']

    # print(np.unique(y)) #012345
    fig, ax = plt.subplots()
    for pose in np.unique(y):
        idx = np.where(y == pose)
        ax.scatter(X[idx,0], X[idx,1], c=colors[pose], label=labels[pose], s=10)
    ax.legend()
    ax.set_title('Use {}'.format(name))
    fig.savefig("{}.png".format(name))


if __name__ == "__main__":
    raw = pd.read_csv('Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(raw)
    y_train_class = np.argmax(y_train.values, axis=1) # Turn one-hot to integer class
    y_test_class = np.argmax(y_test.values, axis=1) # Turn one-hot to integer class

    setups = []
    Hyper_parameters = namedtuple('Hyper_parameters', 'optimizer, learning_rate, epochs, batch_size')
    
    # setups.append(Hyper_parameters( optimizer='Adam',
    #                                 learning_rate=0.02,
    #                                 epochs=200,
    #                                 batch_size=128))
    
    setups.append(Hyper_parameters( optimizer='GradientDescent',
                                    learning_rate=0.01,
                                    epochs=500,
                                    batch_size=64))

    for setup in setups:
        dnn = DNN(features=68, classes=6, optimizer=setup.optimizer, learning_rate=setup.learning_rate)
        loss, loss_validate, acc, acc_validate = dnn.train( X=X_train.values.astype(np.float32),
                                                            y=y_train.values.astype(np.float32),
                                                            X_validate=X_test.values.astype(np.float32),
                                                            y_validate=y_test.values.astype(np.float32),
                                                            epochs=setup.epochs,
                                                            batch_size=setup.batch_size)
        print("Train loss:{}".format(np.asscalar(loss[-1])))
        print("Test loss:{}".format(np.asscalar(loss_validate[-1])))
        fig, ax = plt.subplots()
        ax.plot(range(setup.epochs), loss, label='train')
        ax.plot(range(setup.epochs), loss_validate, label='validate')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('model loss')
        ax.legend()
        # fig.savefig('{}_loss.png'.format(setup.optimizer))

        print("Train accuracy:{}".format(np.asscalar(acc[-1])))
        print("Test accuracy:{}".format(np.asscalar(acc_validate[-1])))
        fig, ax = plt.subplots()
        ax.plot(range(setup.epochs), acc, label='train')
        ax.plot(range(setup.epochs), acc_validate, label='validate')
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title('model acc')
        ax.legend()
        # fig.savefig('{}_accuracy.png'.format(setup.optimizer))

        predicted_class = dnn.predict(X_test)
        precision, recall, f1, micro_prec, micro_recall, micro_f1, macro_prec, macro_recall, macro_f1 = dnn.get_metrics(predicted_class, y_test_class)
        for i in range(6):
            print("Test class {} precision:{}".format(i, precision[i]))
            print("Test class {} recall:{}".format(i, recall[i]))
            print("Test class {} f1-score:{}".format(i, f1[i]))
        print("Test micro average precision:{}".format(micro_prec))
        print("Test micro average recall:{}".format(micro_recall))
        print("Test micro average f1:{}".format(micro_f1))
        print("Test macro average precision:{}".format(macro_prec))
        print("Test macro average recall:{}".format(macro_recall))
        print("Test macro average f1:{}".format(macro_f1))

        # X_hidden = pd.read_csv('Test_no_Ac.csv')
        # X_hidden = (X_hidden - X_hidden.mean()) / X_hidden.std()
        # y_hidden = dnn.predict(X_hidden)
        # f = open("105061110_answer.txt", "w")
        # for idx in range(len(y_hidden)):
        #     f.write("{}\t{}\n".format(idx, y_hidden[idx]+1))
        # f.close()


    # low_dim_X = PCA(n_components=2).fit_transform(X_test.values)
    # plot2d((low_dim_X, y_test_class), name='PCA')

    # low_dim_X = TSNE(n_components=2).fit_transform(X_test.values)
    # plot2d((low_dim_X, y_test_class), name='tSNE')