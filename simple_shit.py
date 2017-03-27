from sklearn import neural_network

#[input,input]
X = [[0,0],[0,1],[1,1],[1,0]]

Y = [False,True,False,True]

mlpclf = neural_network.MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                                      learning_rate_init=0.01, max_iter=50, verbose=True, solver='lbfgs')

mlpclf = mlpclf.fit(X,Y)

prediction_neural = mlpclf.predict([1,1])

print(prediction_neural)
