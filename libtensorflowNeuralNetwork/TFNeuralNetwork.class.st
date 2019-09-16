"
This class represent a neural network build on top of TensorFlow.

This class allow to manage layers and TF session and graph.

This class is on build based of work on libtensorflowpharobindings with respect of Matrix Neural Network of agileartificialintelligence. 

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #TFNeuralNetwork,
	#superclass : #Object,
	#instVars : [
		'graph',
		'session',
		'layers',
		'errors',
		'precisions',
		'loss',
		'expectedLabel',
		'learn',
		'learningRate',
		'batchSize',
		'axis0',
		'one'
	],
	#category : #libtensorflowNeuralNetwork
}

{ #category : #examples }
TFNeuralNetwork class >> exampleFunction [
	"To specify a function for do a regression."

	^ [ :x | 
		| y |
		y := x * 10 - 5.
		y * y sin ].
]

{ #category : #examples }
TFNeuralNetwork class >> exampleNetwork [
	"To build a custom network."

	| net l1 l2 l3 l4 |
	net := self new.
	l1 := TFNeuralLayer new initializeNbOfNeurons: 20 nbOfWeights: 1; activationFunction: ReLU.
	l2 := TFNeuralLayer new initializeNbOfNeurons: 20 nbOfWeights: 20; activationFunction: Sigmoid.
	l3 := TFNeuralLayer new initializeNbOfNeurons: 20 nbOfWeights: 20; activationFunction: Sigmoid.
	l4 := TFNeuralLayer new initializeNbOfNeurons: 1 nbOfWeights: 20; activationFunction: Sigmoid.
	net addLayer: l1;
		addLayer: l2;
		addLayer: l3;
		addLayer: l4;
		fit.
		
	^ net
]

{ #category : #examples }
TFNeuralNetwork class >> exampleTrainedAndPlot [
	"To train a network for learn a function and plot both results of network and true function."

	| function net predictor xValues b ds |
	function := self exampleFunction.
	net := self exampleTrainedOn: function nbEpoch: 10000.
	predictor := [ :x | 
	| result |
	result := net predict: {{x}}.
	result asNumbers first first ].
	xValues := 0 to: 1.0 by: 0.01.
	b := RTGrapher new.
	b extent: 300 @ 200.
	ds := RTData new.
	ds noDot.
	ds points: xValues.
	ds y: predictor.
	ds x: #yourself.
	ds connectColor: Color green.
	b add: ds.
	ds := RTData new.
	ds noDot.
	ds points: xValues.
	ds y: function.
	ds x: #yourself.
	ds connectColor: Color red.
	b add: ds.
	b open.
	^ net
]

{ #category : #examples }
TFNeuralNetwork class >> exampleTrainedOn: function nbEpoch: nbEpoch [
	"To train a custon neural network to learn a function."

	| net rnd interval xs ys |
	net := self exampleNetwork.
	rnd := Random seed: 100.
	xs :=  (1 to: 100) collect: [:i | {rnd next}].
	ys := xs collect: [:x | {function value: x first}].
	interval := 1 to: xs size.
	
	nbEpoch timesRepeat: [
		|x indices y |
		indices := (1 to: 60) collect: [:i| interval atRandom].
		x := indices collect: [:index | xs at: index].
		y :=  indices collect: [:index | ys at: index].
		net predict: x andLearnFrom: y].
	^ net
]

{ #category : #adding }
TFNeuralNetwork >> addLayer: aLayer [
	"To add a layer to the network."

	layers ifNotEmpty: [ 
		layers last nextLayer: aLayer.
		aLayer previousLayer: (layers last) ].
	layers add: aLayer.
	aLayer network: self;
			learningRate: learningRate.
]

{ #category : #accessing }
TFNeuralNetwork >> axis0 [
	"To get axis0 attribute of the network."

	^ axis0
]

{ #category : #accessing }
TFNeuralNetwork >> axis0: anAxis [
	"To set axis0 attribute of the network."

	axis0 := anAxis
]

{ #category : #accessing }
TFNeuralNetwork >> batchSize [
	"To get batch size attribute of the network."

	^ batchSize
]

{ #category : #accessing }
TFNeuralNetwork >> batchSize: aBatchSize [
	"To set batch size attribute of the network."

	batchSize := aBatchSize
]

{ #category : #'as yet unclassified' }
TFNeuralNetwork >> configureInferenceGraph [
	"To configure the inference graph of the layer. Only called at the end of the network configuration."

	| inputs |
	layers doWithIndex: [ :layer :index |
		layer outputs: (graph fromBlock: [:image |
														(index = 1) ifTrue: [ inputs := image ] ifFalse: [ inputs := layer previousLayer outputs ].
														layer inputs: inputs.
														(inputs * (layer weights) + layer biases) sigmoid]
											named: 'layer', index asString) ]
]

{ #category : #'as yet unclassified' }
TFNeuralNetwork >> configureLearningParams [
	"To configure learn parameters of network."

	learn := graph newOperation: 'Identity' named: 'learn' described: [ :description |
		description addInput: loss output.
		layers do: [ :layer |
			description
				addControlInput: layer learnWeights output;
				addControlInput: layer learnBiases output ] ].
]

{ #category : #'as yet unclassified' }
TFNeuralNetwork >> configureLossGraph [
	"To configure the loss graph of the network. Only called at the last layer of the network, in the last configuration."

	loss := graph fromBlock: [ :expected |
														expectedLabel := expected.
														(layers last outputs - expectedLabel) squared meanOn: #(0) asInt32Tensor ]
						inputTypes: {TF_Tensor typeFloat}
						named: 'loss'
]

{ #category : #'as yet unclassified' }
TFNeuralNetwork >> configureParams [
	"To configure weights and baises into the network."

	| inputSize aux  |
	graph fromBlock: [
				layers doWithIndex: [ :layer :index |
					(index = 1) ifTrue: [ inputSize := layer inputSize ] 
										ifFalse: [ inputSize := layer previousLayer numberOfNeurons ].
					aux := graph truncatedNormalRandomShaped: {inputSize. layer numberOfNeurons} stddev: 1.0 / layer inputSize sqrt.
					layer weights: (graph variable: ('weights', index asString) initialValueFrom: aux).
					aux := graph zerosShaped: {layer numberOfNeurons}.
					layer biases: (graph variable: ('biases', index asString) initialValueFrom: aux) ] ]
			named: 'parameters'.
]

{ #category : #accessing }
TFNeuralNetwork >> expectedLabel [
	"To get expectedLabel attribute of the network."

	^ expectedLabel
]

{ #category : #accessing }
TFNeuralNetwork >> expectedLabel: aLabel [
	"To set expectedLabel attribute of the network."

	expectedLabel := aLabel
]

{ #category : #private }
TFNeuralNetwork >> fit [
	"To configure inference, loss and learning graph."

	layers first configureParams.
	layers first configureInferenceGraph.
	self configureLossGraph.
	layers last configureLearningGraph.
	self configureLearningParams.
	self initializeSession
]

{ #category : #accessing }
TFNeuralNetwork >> graph [
	"To get graph attribute of the network."

	^ graph
]

{ #category : #initialization }
TFNeuralNetwork >> initialize [
	"To make some actions before to use the network."

	graph := TF_Graph create.
	layers := OrderedCollection new.
	errors := OrderedCollection new.
	precisions := OrderedCollection new.
	self learningRate: 0.1
]

{ #category : #initialization }
TFNeuralNetwork >> initializeSession [
	"To initialize a session on the created graph."

	session := TF_Session on: graph.
	graph initializeOn: session.
]

{ #category : #accessing }
TFNeuralNetwork >> layers [
	"To get layers attribute of the network."

	^ layers
]

{ #category : #accessing }
TFNeuralNetwork >> learningRate [
	"To get the learning rate of the network."

	^ learningRate
]

{ #category : #accessing }
TFNeuralNetwork >> learningRate: aLearningRate [
	"To set the learning rate of the network."

	learningRate := aLearningRate
]

{ #category : #accessing }
TFNeuralNetwork >> loss [
	"To get loss attribute of the network."

	^ loss
]

{ #category : #accessing }
TFNeuralNetwork >> loss: aLoss [
	"To set loss attribute of the network."

	loss := aLoss
]

{ #category : #accessing }
TFNeuralNetwork >> one [
	"To get one attribute of the network."

	^ one
]

{ #category : #accessing }
TFNeuralNetwork >> one: anOne [
	"To set one attribute of the network."

	one := anOne
]

{ #category : #running }
TFNeuralNetwork >> predict: inputs [
	"To do inference on the network and get the result."

	| results |
	results := session
		runInputs: {layers first inputs input: 0}
		values: {inputs asFloatTensor}
		outputs: {layers last outputs output: 0}.
	^ results first
]

{ #category : #running }
TFNeuralNetwork >> predict: inputs andConpareTo: label [
	"To do inference on the network and compare the result with a label."

	| results |
	results := session
		runInputs: {layers first inputs input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asFloatTensor}
		outputs: {layers last outputs output: 0. loss output: 0}.
	^ results
]

{ #category : #running }
TFNeuralNetwork >> predict: inputs andLearnFrom: label [
	"To do inference on the network and learn suitable parameters."

	| results |
	results := session
		runInputs: {layers first inputs input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asFloatTensor}
		outputs: {loss output: 0 . learn output}.
	^ results
]

{ #category : #accessing }
TFNeuralNetwork >> prediction [
	"To get prediction attribute of the network."

	^ layers last outputs
]

{ #category : #accessing }
TFNeuralNetwork >> session [
	"To get session attribute of the network."

	^ session
]
