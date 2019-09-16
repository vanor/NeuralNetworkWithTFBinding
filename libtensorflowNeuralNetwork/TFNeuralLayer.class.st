"
This class represent a neural layer build on top of TensorFlow. This is a fully connected layer.

This class allow to manage layers and TF session and graph.

This class is on build based of work on libtensorflowpharobindings with respect of Matrix Neural Network of agileartificialintelligence. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.

    Instance Variables
	neurons:		<Object>


    Implementation Points
"
Class {
	#name : #TFNeuralLayer,
	#superclass : #Object,
	#instVars : [
		'weights',
		'biases',
		'nextLayer',
		'previousLayer',
		'learningRate',
		'inputs',
		'outputs',
		'network',
		'sizeOfWeights',
		'numberOfNeurons',
		'backprop',
		'learnWeights',
		'learnBiases',
		'activationFunction'
	],
	#category : #libtensorflowNeuralNetwork
}

{ #category : #accessing }
TFNeuralLayer >> activationFunction: anActivationFunctionClass [
	"To set the activation function of neurons in this layer."

	activationFunction := anActivationFunctionClass
]

{ #category : #accessing }
TFNeuralLayer >> backprop [
	"to get backprop attribute of the layer."

	^ backprop
]

{ #category : #accessing }
TFNeuralLayer >> backprop: aValue [
	"to set backprop attribute of the layer."

	backprop := aValue
]

{ #category : #accessing }
TFNeuralLayer >> biases [
	"To get biases attribute of the layer."

	^ biases
]

{ #category : #accessing }
TFNeuralLayer >> biases: someBiases [
	"To set biases attribute of the layer."

	biases := someBiases
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureInferenceGraph [
	"To configure the inference graph of the layer. Only called at the end of the network configuration."

	outputs := self network graph fromBlock: [:image |
															inputs := image.
															activationFunction activate value: inputs * weights + biases ]
											named: 'layer', (network layers indexOf: self) asString.
	self nextLayer ifNotNil: [ self nextLayer configureInferenceGraphWithPreviousLayer ]
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureInferenceGraphWithPreviousLayer [
	"To link the output of previous layer with input of current layer."

	outputs := self network graph fromBlock: [ inputs := self previousLayer outputs.
															activationFunction activate value: inputs * weights + biases ]
											named: 'layer', (self network layers indexOf: self) asString.
	self nextLayer ifNotNil: [ self nextLayer configureInferenceGraphWithPreviousLayer ]
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureLearningGraph [
	"To configure the learning graph of the network. Only called at the end of the network configuration, for the last layer."

	| learningRate0 count gradient |
	learningRate0 := learningRate asTensor.
	network batchSize: (network graph fromBlock: [(network layers first inputs sizeOn: 0) castTo: TF_Tensor typeFloat] named: 'batchSize').
	network axis0: (network graph const: #(0) asInt32Tensor). 
	network one: (1.0 asTensor asOperationOn: network graph).
	count := network layers indexOf: self.
	
	network graph fromBlock: [ 
							gradient := (network prediction - network expectedLabel).
							learnWeights := weights descent: inputs \* gradient @/ (network batchSize) rate: learningRate0.
							learnBiases := biases descent: (gradient meanOn: network axis0) rate: learningRate0.
							self previousLayer ifNotNil: [ backprop :=  (gradient *\ weights) ] ]
					named: 'learning', count asString.
		
	self previousLayer ifNotNil: [ self previousLayer configureLearningGraphWithNextLayer ].
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureLearningGraphWithNextLayer [
	"To configure the learning graph of the network. For hidden and input layers."

	| count |
	count := self network layers indexOf: self.
	
	network graph
		fromBlock: [ activationFunction updateNetwork: network onLayer: self ]
		named: 'learning', count asString.
		
	self previousLayer ifNotNil: [ self previousLayer configureLearningGraphWithNextLayer ]
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureLossGraph [
	"To configure the loss graph of the network. Only called at the last layer of the network, in the last configuration."

	self network loss: (self network graph fromBlock: [ :expected |
														self network expectedLabel: expected.
														(outputs - self network expectedLabel) squared meanOn: #(0) asInt32Tensor ]
									inputTypes: {TF_Tensor typeFloat}
									named: 'loss')
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureParams [
	"To configure weights and baises into the first layer of the network."

	| inputSize aux count |
	count := self network layers indexOf: self.
	inputSize := self inputSize.
	self network graph fromBlock: [
									aux := self network graph truncatedNormalRandomShaped: {inputSize . numberOfNeurons} stddev: 1.0 / inputSize sqrt.
									weights := self network graph variable: ('weights', count asString) initialValueFrom: aux.
									aux := self network graph zerosShaped: {numberOfNeurons}.
									biases := self network graph variable: ('biases', count asString) initialValueFrom: aux ]
							named: 'parameters', count asString.
	
	self nextLayer ifNotNil: [ self nextLayer configureParamsWithPreviousLayer ]
]

{ #category : #'as yet unclassified' }
TFNeuralLayer >> configureParamsWithPreviousLayer [
	"To configure weights and baises into hidden and output layers of the network."

	| inputSize aux count |
	count := self network layers indexOf: self.
	inputSize := self previousLayer numberOfNeurons.
	self network graph fromBlock: [
									aux := self network graph truncatedNormalRandomShaped: {inputSize . numberOfNeurons} stddev: 1.0 / inputSize sqrt.
									weights := self network graph variable: ('weights', count asString) initialValueFrom: aux.
									aux := self network graph zerosShaped: {numberOfNeurons}.
									biases := self network graph variable: ('biases', count asString) initialValueFrom: aux ]
							named: 'parameters', count asString.
	
	self nextLayer ifNotNil: [ self nextLayer configureParamsWithPreviousLayer ]
]

{ #category : #accessing }
TFNeuralLayer >> initializeNbOfNeurons: nbOfNeurons nbOfWeights: nbOfWeights [
	"To set parameters into the layer. It is the main method to initialize the layer."

	sizeOfWeights := nbOfWeights.
	numberOfNeurons := nbOfNeurons
]

{ #category : #accessing }
TFNeuralLayer >> inputSize [
	"To get the input size of the layer."

	^ sizeOfWeights
]

{ #category : #accessing }
TFNeuralLayer >> inputs [
	"To get the inputs of the layer."

	^ inputs
]

{ #category : #accessing }
TFNeuralLayer >> inputs: someInputs [
	"To set the inputs of the layer."

	inputs := someInputs
]

{ #category : #accessing }
TFNeuralLayer >> learnBiases [
	"To get the learnBiases attribute of the current layer."

	^ learnBiases
]

{ #category : #accessing }
TFNeuralLayer >> learnBiases: someValues [
	"To set the learnBiases attribute of the current layer."

	learnBiases := someValues
]

{ #category : #accessing }
TFNeuralLayer >> learnWeights [
	"To get the learnWeights attribute of the current layer."

	^ learnWeights
]

{ #category : #accessing }
TFNeuralLayer >> learnWeights: someValues [
	"To set the learnWeights attribute of the current layer."

	learnWeights := someValues
]

{ #category : #accessing }
TFNeuralLayer >> learningRate [
	"To get the learning rate of the current layer."

	^ learningRate
]

{ #category : #accessing }
TFNeuralLayer >> learningRate: aLearningRate [
	"To set the learning rate of the current layer."

	learningRate := aLearningRate
]

{ #category : #accessing }
TFNeuralLayer >> network [
	"To get network attribute of the current layer."

	^ network
]

{ #category : #accessing }
TFNeuralLayer >> network: aNetwork [
	"To set network attribute of the current layer."

	network := aNetwork
]

{ #category : #accessing }
TFNeuralLayer >> nextLayer [
	"To get the next layer of the current layer."

	^ nextLayer
]

{ #category : #accessing }
TFNeuralLayer >> nextLayer: aLayer [
	"To set the next layer of the current layer."

	nextLayer := aLayer.
]

{ #category : #accessing }
TFNeuralLayer >> numberOfNeurons [
	"To get the number of neurons in the current layer."

	^ numberOfNeurons
]

{ #category : #accessing }
TFNeuralLayer >> outputs [
	"To get the outputs of the layer."

	^ outputs
]

{ #category : #accessing }
TFNeuralLayer >> outputs: someOutputs [
	"To set the outputs of the layer."

	outputs := someOutputs
]

{ #category : #accessing }
TFNeuralLayer >> previousLayer [
	"To get the previous layer of the current layer."

	^ previousLayer
]

{ #category : #accessing }
TFNeuralLayer >> previousLayer: aLayer [
	"To set the previous layer of the current layer."

	previousLayer := aLayer.
]

{ #category : #accessing }
TFNeuralLayer >> weights [
	"To get weights attribute of the layer."

	^ weights
]

{ #category : #accessing }
TFNeuralLayer >> weights: someWeights [
	"To set weights attribute of the layer."

	weights := someWeights
]
