"
This class represent the ReLU activation function.
"
Class {
	#name : #ReLU,
	#superclass : #ActivationFunction,
	#category : #libtensorflowNeuralNetwork
}

{ #category : #'event handling' }
ReLU class >> activate [
	"To compute activation result of neurons."

	^ [ :x | x rectified ]
]

{ #category : #'event handling' }
ReLU class >> derivative [
	"To compute derivative of neurons for backpropagation."

	^ [ :nextLayerBackprop :outputs | nextLayerBackprop timesRectifiedGradOf: outputs ]
]

{ #category : #'event handling' }
ReLU class >> updateNetwork: network onLayer: layer [
	"To update weights and biases of the specified layer."

	| gradient learningRate |
	learningRate := network learningRate asTensor.
	gradient := self derivative value: (layer nextLayer backprop) value: layer outputs.
	layer learnWeights: (layer weights descent: layer inputs \* gradient rate: learningRate).
	layer learnBiases: (layer biases descent: (gradient meanOn: network axis0) rate: learningRate).
	layer previousLayer ifNotNil: [ layer backprop: (gradient *\ layer weights) ]
]
