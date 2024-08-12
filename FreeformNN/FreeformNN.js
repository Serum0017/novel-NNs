class FreeformNN {
    // hyperparameters:
    // physicalBrainSize: size of the grid on which to simulate the brain. Brain is a circle in 2d. Could try higher dimensions.
    // neuronConnectionRange: how far apart to connect the neurons. PhysicalBrainSize must be a multiple of neuronConnectionRange.
    // num inputs: number of inputs for the task
    // num outputs: number of outputs
    // steps: amount of times to simulate the graph algorithm before measuring outputs and terminating
    // learning rate: standard NN learning rate
    constructor({
        physicalBrainSize,
        neuronConnectionRange=spatialHashSize/10,
        numInputs,
        numOutputs,
        steps=10,
        learningRate=0.01
    }){
        // all nodes are static
        this.nodes = [];
        this.spHash = new SpatialHash(physicalBrainSize, neuronConnectionRange);

        this.gradients = [];

        this.size = physicalBrainSize;

        // [minCoord, maxCoord)
        this.minCoord = this.neuronConnectionRange;
        this.maxCoord = this.physicalBrainSize - this.neuronConnectionRange;

        this.coordinateRadius = (this.maxCoord - this.minCoord) / 2;
        this.coordinateMiddle = this.physicalBrainSize / 2;

        // this.numInputs = numInputs;
        // this.numOutputs = numOutputs;

        // create inputs and outputs
        for(let i = 0; i < numInputs+numOutputs; i++){
            this.nodes.push(new Neuron(this.randomNeuronPosition()));
        }

        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        this.numSteps = steps;

        this.newOutputs = [];

        this.learningRate = learningRate;

        // for derivative approximation
        this.h = 0.01;
    }

    forward(inputs=[]){
        // reset all outputs
        for(let i = 0; i < this.nodes.length; i++){
            this.nodes[i].output = 0;
        }

        // set inputs. Inputs are the first section of the array
        for(let i = 0; i < inputs.length; i++){
            this.nodes[i].output = inputs[i];// first layer doesn't apply activation function, it just feeds in inputs
        }

        this.newOutputs = new Array(this.nodes.length);

        // every step, propogate the inputs by first calculating them based on exiting and then updating
        for(let i = 0; i < steps.length; i++){
            for(let j = 0; j < this.nodes.length; j++){
                this.newOutputs[j] = this.nodes[j].calculateOutput();
            }

            for(let j = 0; j < this.nodes.length; j++){
                this.nodes[j].output = this.newOutputs[j];
            }
        }

        // calculating outputs of the output nodes
        const outputs = [];
        for(let i = this.numInputs; i < this.numInputs+this.numOutputs; i++){
            outputs.push(this.nodes[i].output);
        }
        return outputs;
    }

    // data = array of [inputs1, inputs2, ...], outputs are the same
    train(inputs, outputs){
        // goal:
        // split data up into epochs, train, gradient descent, etc.

        // for now: let's do the slow approach to test that it is working
        // for every neuron
        let gradients = [];
        let biasGradients = [];
        for(let i = 0; i < this.nodes.length; i++){
            // for all data
            let error = 0;
            for(let j = 0; j < inputs.length; j++){
                // calculate the error rn
                error += this.forward(inputs[j]) - outputs[j];
            }

            // calculate what happens if we change each weight by a tiny bit
            gradients[i] = [];
            for(let j = 0; j < this.nodes[i].weights.length; j++){
                this.nodes[i].weights[j] += this.h;
                let gradientError = 0;
                for(let j = 0; j < inputs.length; j++){
                    gradientError += this.forward(inputs[j]) - outputs[j];
                }
                this.nodes[i].weights[j] -= this.h;

                // approximation of the derivative
                gradients[i][j] = (gradientError - error) / this.h;
            }

            // calculate what happens if we change the bias by a tiny bit
            this.nodes[i].bias += this.h;
            let biasError = 0;
            for(let j = 0; j < inputs.length; j++){
                biasError += this.forward(inputs[j]) - outputs[j];
            }
            this.nodes[i].bias -= this.h;

            biasGradients.push((biasError - error) / this.h)
        }

        // apply gradients
        for(let i = 0; i < this.nodes.length; i++){
            for(let j = 0; j < gradients[i].length; j++){
                this.nodes[i].weights[j] -= gradients[i][j] * this.learningRate;
            }
            this.nodes[i].bias -= biasGradients[i] * this.learningRate;
        }

        // TODO: random mutations like creating/removing/moving neurons
    }

    fit(data){
        // update gradients with all the data

    }

    backPropogate(){

    }

    // uniformly random within a circle
    randomNeuronPosition(){
        const r = Math.sqrt(Math.random()) * this.coordinateRadius;
        const theta = Math.random() * Math.PI * 2;
        return {
            x: Math.cos(theta) * r + this.coordinateMiddle,
            y: Math.sin(theta) * r + this.coordinateMiddle
        }
    }
}