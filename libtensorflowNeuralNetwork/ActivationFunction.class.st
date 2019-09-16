"
This class is an abstract class which represent the activation function of one layer.

This class is responsible of the computation of activation and derivative values.

With respect of work in AgileArtificial book.

Public API and Key Messages

- message one   
- message two 
- (for bonus points) how to create instances.

   One simple example is simply gorgeous.
 
Internal Representation and Key Implementation Points.


    Implementation Points
"
Class {
	#name : #ActivationFunction,
	#superclass : #Object,
	#category : #libtensorflowNeuralNetwork
}

{ #category : #'event handling' }
ActivationFunction class >> activate [
	"To compute the value of activation of neurons."

	self subclassResponsibility.
]

{ #category : #'event handling' }
ActivationFunction class >> derivative [
	"To compute derivative of neurons."

	self subclassResponsibility.
]

{ #category : #'event handling' }
ActivationFunction class >> updateNetwork: network onLayer: layer [
	"To update weights and biases of the specified layer."

	self subclassResponsibility.
]
