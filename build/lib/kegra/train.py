from __future__ import print_function
# 注意：conda环境为FL
from keras.layers import Input, Dropout, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from layers.graph import GraphConvolution
from utils import *

import time
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(4)
# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 20000
PATIENCE = 300  # early stopping patience

# Get data
X, A1,A2,A3, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A1_ = preprocess_adj(A1, SYM_NORM)
    A2_ = preprocess_adj(A2, SYM_NORM)
    A3_ = preprocess_adj(A3, SYM_NORM)
    support = 1
    graph = [X, A1,A2,A3]
    G1 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    G2 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    G3 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
# H = Dropout(0.1)(X_in)
H = X_in
H1 = GraphConvolution(16, 1,support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G1+G2+G3)
# H1 = GraphConvolution(8, 1,support, activation='relu', kernel_regularizer=l2(5e-4))([H1]+G1+G2+G3)
H2 = GraphConvolution(16, 2,support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G1+G2+G3)
H3 = GraphConvolution(16, 3,support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G1+G2+G3)

# H2 = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G)
# H3 = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G)

# H = Dropout(0.1)(H)
Y1 = GraphConvolution(y.shape[1],1 ,support)([H1]+G1+G2+G3)
Y2 = GraphConvolution(y.shape[1],2, support)([H2]+G1+G2+G3)
Y3 = GraphConvolution(y.shape[1],3, support)([H3]+G1+G2+G3)

Y = Concatenate()([Y1,Y2,Y3])

Y = Dense(y.shape[1])(Y1)


# Compile model
model = Model(inputs=[X_in]+G1+G2+G3, outputs=Y)

model.compile(loss='mse', optimizer=Adam(lr=0.001))
print(model.summary())
# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH+1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A1.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A1.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
