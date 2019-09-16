"
This class represent the sigmoid activation function.
"
Class {
	#name : #Sigmoid,
	#superclass : #ActivationFunction,
	#category : #libtensorflowNeuralNetwork
}

{ #category : #'event handling' }
Sigmoid class >> activate [
	"To compute activation result of neurons."

	^ [ :x | x sigmoid ]
]

{ #category : #'event handling' }
Sigmoid class >> derivative [
	"To compute derivative of neurons for backpropagation."

	^ [ :nextLayerBackprop :outputs :one | nextLayerBackprop @* outputs @* (one - outputs) ]
]

{ #category : #'event handling' }
Sigmoid class >> updateNetwork: network onLayer: layer [
	"To update weights and biases of the specified layer."

	| gradient learningRate |
	learningRate := network learningRate asTensor.
	gradient := self derivative value: (layer nextLayer backprop) value: (layer outputs) value: network one.
	layer learnWeights: (layer weights descent: layer inputs \* gradient @/ (network batchSize) rate: learningRate).
	layer learnBiases: (layer biases descent: (gradient meanOn: network axis0) rate: learningRate).
	layer previousLayer ifNotNil: [ layer backprop: (gradient *\ layer weights) ]
]
