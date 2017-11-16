'''
Code to attempt ordinal inference in TensorFlow.

This appears not to be possible in Sklearn because you can't specify the cost function, but
in TensorFlow we specify the objective when generating the gradient descent optimizer. The 
idea is this: make rows in the training set that are not decreasing on both sides from the
maximum cost Nx as much as one that is so.

Worry: that the learning will get stuck.

Toy dataset: 
4 categories
Categories 0 and 3 have one bit as high probability
Categories 2 and 3 have one bit as high probability
Bits are randomly reassigned some fraction of the time.

Worry: ordinality plays no role in this data set. 

Lessons/questions/improvements
1. Segregate reporting -- make it easier to switch on or off (unless tf.summary can already do 
this ... I do not see any such option).
2. Segregate graph better. It's ok, but make it more straightforward to replace.
3. Make training more straightforward to replace. 
'''

from time import time
import datetime

import abc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf


class DataGenerator:

    def __init__(self,
                 ncats=4,
                 cat_dist=(25, 25, 25, 25),
                 p0=(.8, .2, .15, .8),
                 p1=(.1, .05, .75, .85),
                 flip_prob=.05):
        self.ncats = ncats
        self.cat_dist = np.array(cat_dist)
        self.cat_dist = self.cat_dist / self.cat_dist.sum()
        self.p0 = p0
        self.p1 = p1
        self.flip_prob = flip_prob

    def point_state(self):
        return np.random.choice(range(self.ncats), p=self.cat_dist)

    def genpoint(self):
        cat = self.point_state()
        s = {'cat': cat,
             'b0': 2 * (int(np.random.random() < self.p0[cat]) - .5),
             'b1': 2 * (int(np.random.random() < self.p1[cat]) - .5)}
        if np.random.random() < self.flip_prob:
            s['cat'] = self.point_state()
        return s

    def generate_points(self, npoints=10000):
        return pd.DataFrame([self.genpoint() for _ in range(npoints)])

    def generate_train_test(self, frac_train=0.7, npoints=10000):
        df = self.generate_points(npoints)
        train = df.sample(frac=frac_train)
        test = df[~df.index.isin(train.index)].reset_index(drop=True)
        train = train.reset_index(drop=True)
        return train, test

    def onehot(self, df):
        '''
        Converts dataframe with 'cat' into a one-hot split dataset
        '''
        X = df.drop('cat', axis=1).values
        eye = np.eye(df.cat.max() + 1)
        y = eye[df.cat.values]
        return X, y


class SimpleFitter:
    '''
    Just do classification using standard sklearn stuff

    Unfortunately sklearn cannot predict multidimensionally by default

    Maybe I can do 4 fits?
    '''

    def __init__(self):
        pass

    def fit_rf(self, X, y, *args):
        return self.fit(X, y, RandomForestClassifier(*args))

    def fit_nn(self, X, y, *args):
        return self.fit(X, y, MLPClassifier(*args))

    def fit(self, X, y, model):
        model.fit(X, y)
        return model

    def evaluate(self, X, y, model):
        pred = model.predict(X)
        return pd.DataFrame((y - pred)**2).groupy(0).size().sort_index()


class TFFitter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def costfunction(self):
        pass

    @abc.abstractmethod
    def xinit(self, *args, **kwargs):
        pass

    def setup_reports_special(self):
        pass

    def __init__(self,
                 nhiddenlayers,
                 layersize,
                 nfeatures,
                 nlabels,
                 opt_step=1e-5,
                 noise=0.01,
                 train_frac=0.5,
                 report_frac=0.1,
                 savedir=None,
                 optimizer=None,
                 include_bias=False,
                 zfunc=None,
                 tblog=True):

        self.nhiddenlayers = nhiddenlayers
        self.layersize = layersize
        self.nfeatures = nfeatures
        self.nlabels = nlabels
        self.opt_step = opt_step
        self.noise = noise
        self.train_frac = train_frac
        self.report_frac = report_frac
        self.savedir = savedir or 'save_{}_{}'.format(self.__class__.__name__,
                                                      datetime.datetime.now().strftime('%y%m%d_%H%M%S_%f'))
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer(self.opt_step)
        self.include_bias = include_bias
        self.zfunc = zfunc or tf.sigmoid
        self.tblog = tblog

    def _generate_session(self):
        return tf.Session()

    def _generate_varib(self, shape, name, noise=None):
        noise = noise or self.noise
        v = tf.Variable(tf.random_normal(shape, stddev=self.noise), name=name)
        if self.tblog:
            tf.summary.scalar(v.op.name, tf.reduce_mean(v))
            tf.summary.scalar('{}-nan'.format(v.op.name), tf.reduce_mean(tf.cast(tf.is_nan(v), tf.float32)))
            tf.summary.histogram(v.op.name, v)
        return v

    def _generate_b(self, shape, name):
        if self.include_bias:
            return self._generate_varib(shape, name, noise=self.noise / 1e5)
        return 0

    def _generate_W(self, shape, name):
        return self._generate_varib(shape, name)

    def _generate_z(self, m1, m2, b, name):
        z = self.zfunc(tf.matmul(m1, m2) + b, name=name)
        if self.tblog:
            tf.summary.scalar(z.op.name, tf.reduce_mean(z))
            tf.summary.histogram(z.op.name, z)
        return z

    def generate_graph_and_variables(self):
        self.xx = tf.placeholder(tf.float32, [None, self.nfeatures], name='xx')

        # input layer
        bs = [self._generate_b([self.layersize], 'b0'), ]
        Ws = [self._generate_W([self.nfeatures, self.layersize], 'W0'), ]
        zs = [self._generate_z(self.xx, Ws[-1], bs[-1], 'z0'), ]

        # hidden layer
        for i in range(1, self.nhiddenlayers + 1):
            bs.append(self._generate_b([self.layersize], 'b{}'.format(i)))
            Ws.append(self._generate_W([self.layersize, self.layersize], 'W{}'.format(i)))
            zs.append(self._generate_z(zs[-1], Ws[-1], bs[-1], 'z{}'.format(i)))

        # output layer
        bs.append(self._generate_b([self.nlabels], 'b{}'.format(self.nhiddenlayers + 1)))
        Ws.append(self._generate_W([self.layersize, self.nlabels], 'W{}'.format(self.nhiddenlayers + 1)))
        yy = tf.matmul(zs[-1], Ws[-1], name='yy') + bs[-1]

        self.bs = bs
        self.Ws = Ws
        self.zs = zs
        self.yy = yy
        self.yy_ = tf.placeholder(tf.float32, [None, self.nlabels], name='yy_')

        self.sess = self._generate_session()

    def sample_data(self, X, y, frac):
        nsel = int(X.shape[0] * frac)
        p = np.random.randint(X.shape[0], size=nsel)
        return X[p], y[p]

    def definestuff(self, X, y):
        self.trainx, self.trainy = self.sample_data(X, y, self.train_frac)
        self.generate_graph_and_variables()

    # Overrideable report function
    def define_argmax_reports(self):
        am_yy_ = tf.argmax(self.yy_, 1)
        am_yy = tf.argmax(self.yy, 1)

        argmax_equal = tf.cast(tf.equal(am_yy, am_yy_), tf.float32, name='argmax_equal')
        self.mean_argmax_equal = tf.reduce_mean(argmax_equal, name='mean_argmax_equal')
        self.histogram(argmax_equal)

        self.rmsd_argmax = tf.sqrt(tf.reduce_mean(tf.cast(tf.pow(am_yy - am_yy_, 2),
                                                          tf.float32)),
                                   name='rmsd_argmax')
        self.scalar(self.rmsd_argmax)

    def define_msd_reports(self):
        msd_tensor = tf.reduce_mean(tf.pow(tf.nn.softmax(self.yy) - self.yy_, 2), name='msd_tensor')
        self.rmsd_tensor = tf.sqrt(tf.cast(msd_tensor, tf.float32), name='rmsd_tensor')
        self.histogram(msd_tensor)
        self.scalar(self.rmsd_tensor)

    def histogram(self, f):
        if self.tblog:
            tf.summary.histogram(f.op.name, f)

    def scalar(self, f):
        if self.tblog:
            tf.summary.scalar(f.op.name, f)

    def setup_reports(self):
        # setup reporting and TensorBoard outputs

        ### Argmax metrics ###
        # These deal with the category assignment
        self.define_argmax_reports()

        ### Tensor metrics ###
        # These deal with the full output tensor, i.e., comparing [0 0 1] with [0 .5 .5]
        self.define_msd_reports()

        ### Cost ###
        # There was a good reason why I put this here
        self.cost = self.costfunction()
        self.scalar(self.cost)

        self.setup_reports_special()

    def print_report(self, item, feed_dict):
        print('\t{} = {}'.format(item.op.name,
                                 self.sess.run(item, feed_dict)))

    def train(self, X, y, nsteps=10000, printfreq=500):
        self.definestuff(X, y)
        self.setup_reports()

        if self.tblog:
            summaries = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.savedir, self.sess.graph)

        # objective function
        opt = self.optimizer.minimize(self.cost)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        t = time()

        for step in range(nsteps):
            self.trainx, self.trainy = self.sample_data(X, y, self.train_frac)

            if step % printfreq == 0 or step == nsteps - 1:
                reportx, reporty = self.sample_data(X, y, self.report_frac)
                print('Step {}'.format(step))
                fd = {self.xx: reportx, self.yy_: reporty}
                self.print_report(self.mean_argmax_equal, fd)
                self.print_report(self.rmsd_argmax, fd)
                self.print_report(self.rmsd_tensor, fd)
                self.print_report(self.cost, fd)
            if self.tblog:
                summary_writer.add_summary(self.sess.run(summaries,
                                                         feed_dict={self.xx: self.trainx,
                                                                    self.yy_: self.trainy}), step)
            self.sess.run(opt, feed_dict={self.xx: self.trainx, self.yy_: self.trainy})

        if self.tblog:
            summary_writer.flush()

        print('Run took {:.2f} seconds'.format(time() - t))

    # TODO:
    def save(self):
        pass

    # TODO:
    def predict(self, X):
        pass

    # TODO:
    def analyze(self, X):
        pass

# TODO:


def load():
    pass


class OrdinalFitter0(TFFitter):
    '''
    Cost function is straight cross entropy
    '''

    def xinit(self):
        pass

    def costfunction(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.yy_, logits=self.yy), name='cost')


class OrdinalFitter1(TFFitter):
    '''
    A fitter based on TensorFlow that penalizes rows that do not decrease from the maximum
    '''

    def xinit(self,
              non_decrease_cost_factor=3):
        self.ndcf = non_decrease_cost_factor
        return self

    # TODO:
    def costfunction(self):

        costfactor = self.costfactor()
        cearr = tf.nn.softmax_cross_entropy_with_logits(labels=self.yy_, logits=self.yy)
        return tf.reduce_mean(cearr * costfactor, name='cost')

    def costfactor(self):
        '''
        We take two vectors:
        - whether the output row is increasing (e.g., [1 2 3 5 4] -> [1 1 1 0])
        - whether the value is after the maximum (e.g., [1 2 3 5 4] -> [ 0 0 0 1]
        The xor of this vector will be all 1's if the vector has a maximum and all 
        other values decrease monotonically from the left and right of there. The
        average of xor is a number between 1 (conforming) and 0 (nuts). We then use
        average xor in an exponential that penalizes most at 0 and least at 1.
        '''
        # increasing
        drop_first_col = tf.slice(self.yy, [0, 1], [-1, -1])
        drop_last_col = tf.slice(self.yy, [0, 0], [-1, self.nlabels - 1])
        increasing = tf.greater(drop_first_col, drop_last_col)

        # after max
        zz = tf.Variable(list(range(self.nlabels)))
        am = tf.expand_dims(tf.cast(tf.argmax(self.yy, 1), tf.int32), 1)
        above_max = tf.slice(tf.greater_equal(zz, am), [0, 0], [-1, self.nlabels - 1])

        # xor
        xor = tf.cast(tf.logical_xor(above_max, increasing), tf.float32)
        rowmean = tf.reduce_mean(xor, 1)
        return self.penfunc(rowmean)

    def penfunc(self, v):
        return 1 + (self.ndcf - 1) * tf.exp(-self.ndcf * v)

    def setup_reports_special(self):
        cf = self.costfactor()
        if self.tblog:
            tf.summary.scalar('mean_cf', tf.reduce_mean(cf))
            tf.summary.histogram('cf', cf)


class OrdinalFitter1b(OrdinalFitter1):
    '''
    Version of OF1 with a linear cost function. This should penalize nondecreasing
    outputs more strongly.
    '''

    def penfunc(self, v):
        return self.ndcf + (1 - self.ndcf) * v


class OrdinalFitter2(TFFitter):
    '''
    A fitter based on TensorFlow that uses the method from this paper for ordinal regression:
    https://arxiv.org/pdf/0704.1028.pdf
    A Neural Network Approach to Ordinal Regression
    Jianlin Cheng
    Journal???
    2007

    Training set labels must be converted from one-hot to "cumulative-hot" (?):

    [0 0 1 0] -> [1 1 1 0]

    Cross entropy can be calculated still, as long as it's with the sigmoid rather than softmax
    function.

    To compute accuracy, however, we have to convert back:

    [1 1 1 0] -> [0 0 1 0]

    Finally, there is an arbitrary decision to be made when computing accuracy. Cheng says the
    output of the sigmoid can be compared to 0.5, which seems pretty reasonable. 
    '''

    def xinit(self, inclass_cutoff=0.5):
        self.inclass_cutoff = inclass_cutoff

    def costfunction(self):
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yy_,
                                                     logits=self.yy)
        return tf.reduce_mean(ce, name='cost')

    def train(self, X, y, *args, **kwargs):
        # requires messing with the y data
        print('Initial y =\n{}'.format(y[:10].astype(int)))
        y = np.bitwise_or(np.greater_equal(np.argmax(y, axis=1).reshape(-1, 1),
                                           range(y.shape[1])).astype(int),
                          y.astype(int))
        print('Training y =\n{}'.format(y[:10]))
        super().train(X, y, *args, **kwargs)

    def convert_output(self, y):
        # TODO: change this to use the 'unconvert' function
        '''
        Given y data in a cumulative form ([1 1 1 0 ...]) convert back to 
        non-cumulative ([0 0 1 0 ... ])

        y has already been evaluated from the model but it has not been
        converted through the sigmoid
        '''

        yest = 1 / (1 + np.exp(-y))
        # This is cheating, but it works
        fudge = np.linspace(0, 1, self.nlabels) / 10
        yest = ((yest > self.inclass_cutoff).astype(float) + fudge).argmax(axis=1)
        yest = np.eye(self.nlabels)[yest]
        return yest

    def unconvert(self):
        fudge = np.linspace(0, 1, self.nlabels) / 10
        yest = tf.argmax(tf.cast(tf.sigmoid(self.yy) > self.inclass_cutoff, tf.float32) + fudge, 1)
        ytrue = tf.argmax(self.yy_ + fudge, 1)
        return yest, ytrue

    # overrides parent because argmax can't be interpreted the same
    def define_argmax_reports(self):
        yest, ytrue = self.unconvert()

        argmax_equal = tf.cast(tf.equal(yest, ytrue), tf.float32, name='argmax_equal')
        self.mean_argmax_equal = tf.reduce_mean(argmax_equal, name='mean_argmax_equal')
        self.histogram(argmax_equal)

        self.rmsd_argmax = tf.sqrt(tf.reduce_mean(tf.cast(tf.pow(yest - ytrue, 2),
                                                          tf.float32)),
                                   name='rmsd_argmax')
        self.scalar(self.rmsd_argmax)

    # overrides parent
    def define_msd_reports(self):
        yest, ytrue = self.unconvert()
        eye = tf.eye(self.nlabels)
        yest = tf.gather(eye, yest)
        ytrue = tf.gather(eye, ytrue)

        msd_tensor = tf.reduce_mean(tf.cast(tf.pow(yest - ytrue, 2),
                                            tf.float32),
                                    name='msd_tensor')
        self.rmsd_tensor = tf.sqrt(msd_tensor, name='rmsd_tensor')
        self.histogram(msd_tensor)
