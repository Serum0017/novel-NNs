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
        neuronConnectionRange=physicalBrainSize/6,
        numInputs,
        numOutputs,
        steps=10,
        learningRate=0.01,
        createNeuronProbability=0.0008,
        // connectionRate=1,

        initialGridSize=.3,

        // takes up .8 of the entire grid
        initialSpread=0.8,

        gridRandomOffsetMagnitude=.01,

        connectionChance=0.5,
        doubleConnectionChance=0.5,
    }){
        // all nodes are static
        this.nodes = [];
        this.spHash = new SpatialHash(physicalBrainSize, neuronConnectionRange);

        this.gradients = [];

        this.size = physicalBrainSize;
        this.neuronConnectionRange = neuronConnectionRange;

        // [minCoord, maxCoord)
        this.minCoord = this.neuronConnectionRange;
        this.maxCoord = this.size - this.neuronConnectionRange;

        this.coordinateRadius = (this.maxCoord - this.minCoord) / 2;
        this.coordinateMiddle = this.size / 2;
        
        this.connectionChance = connectionChance;
        this.doubleConnectionChance = doubleConnectionChance;

        // this.numInputs = numInputs;
        // this.numOutputs = numOutputs;

        // create inputs and outputs equally spaced on the edge
        for(let i = 0; i < numInputs; i++){
            this.addNeuron({
                x: this.coordinateMiddle - this.coordinateRadius * .99,
                y: this.coordinateMiddle + (numInputs === 1 ? 0 : (i / (numInputs-1)  - .5)) * this.coordinateRadius * 2 * .99,
            }).isInput = true;
        }

        for(let i = 0; i < numOutputs; i++){
            const neuron = this.addNeuron({
                x: this.coordinateMiddle + this.coordinateRadius * .99,
                y: this.coordinateMiddle + (numOutputs === 1 ? 0 : (i / (numOutputs-1) - .5)) * this.coordinateRadius * 2 * .99,
            })
            neuron.isOutput = true;
            // for(let j = 0; j < numInputs; j++){
            //     neuron.addConnection(this.nodes[j]);
            // }
        }

        // for(let i = 0; i < initialHiddenNeurons.length; i++){
        //     this.addNeuron({
        //         x: this.coordinateMiddle + initialHiddenNeurons[i][0] * this.coordinateRadius,
        //         y: this.coordinateMiddle + initialHiddenNeurons[i][1] * this.coordinateRadius
        //     })
        // }

        // for(let i = 0; i < initialNetworkConnections.length; i++){
        //     this.nodes[initialNetworkConnections[i][1]].addConnection(this.nodes[initialNetworkConnections[i][0]]);
        // }

        for(let x = -1; x < 1; x += initialGridSize * 2){
            for(let y = -1; y < 1; y += initialGridSize * 2){
                this.addNeuron({
                    x: this.coordinateMiddle + x * this.coordinateRadius * initialSpread + (Math.random()*2-1) * gridRandomOffsetMagnitude,
                    y: this.coordinateMiddle + y * this.coordinateRadius * initialSpread + (Math.random()*2-1) * gridRandomOffsetMagnitude,
                })
            }
        }

        for(let i = numInputs+numOutputs; i < this.nodes.length; i++){
            this.connectToNearby(this.nodes[i]);
        }

        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        this.numSteps = steps;

        this.newOutputs = [];

        this.learningRate = learningRate;

        this.createNeuronProbability = createNeuronProbability;

        // this.connectionRate = connectionRate;

        // for derivative approximation
        this.h = 0.0001;
    }

    addNeuron(pos=this.randomNeuronPosition()){
        this.nodes.push(new Neuron(pos));
        this.spHash.addEntity(this.nodes[this.nodes.length-1]);
        return this.nodes[this.nodes.length-1];
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

        // console.log(this.nodes.map(n => n.output));

        this.newOutputs = new Array(this.nodes.length);

        // every step, propogate the inputs by first calculating them based on exiting and then updating
        for(let i = 0; i < this.numSteps; i++){
            for(let j = 0; j < this.nodes.length; j++){
                this.newOutputs[j] = this.nodes[j].calculateOutput();
            }

            // console.log(this.newOutputs);

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
    train(inputs=[[]], outputs=[[]]){
        // goal:
        // split data up into epochs, train, gradient descent, etc.

        // for now: let's do the slow approach to test that it is working
        // for every neuron
        let gradients = [];
        let biasGradients = [];

        // find the error
        let error = 0;
        for(let i = 0; i < inputs.length; i++){
            error += (this.forward(inputs[i]) - outputs[i]) ** 2;

            // console.log(inputs[i], this.forward(inputs[i]), outputs[i]);
        }

        // console.log({error});

        for(let i = 0; i < this.nodes.length; i++){
            // calculate what happens if we change each weight by a tiny bit
            gradients[i] = [];
            for(let j = 0; j < this.nodes[i].weights.length; j++){
                this.nodes[i].weights[j] += this.h;
                let gradientError = 0;
                for(let j = 0; j < inputs.length; j++){
                    gradientError += (this.forward(inputs[j]) - outputs[j]) ** 2;
                }
                this.nodes[i].weights[j] -= this.h;

                // approximation of the derivative
                gradients[i][j] = (gradientError - error) / this.h;
            }

            // calculate what happens if we change the bias by a tiny bit
            this.nodes[i].bias += this.h;
            let biasError = 0;
            for(let j = 0; j < inputs.length; j++){
                biasError += (this.forward(inputs[j]) - outputs[j]) ** 2;

                // console.log(this.nodes[i].bias, inputs[j], outputs[j], {forward: this.forward(inputs[j])});
            }
            this.nodes[i].bias -= this.h;

            biasGradients[i] = (biasError - error) / this.h;

            // console.log({biasError});

            // if(this.nodes[i].isOutput)console.log(biasError, error, this.nodes[i].bias);

            // if(this.nodes[i].isOutput === true)console.log(biasGradients[i], biasError, error, this.h);
        }

        // apply gradients
        for(let i = 0; i < this.nodes.length; i++){
            for(let j = 0; j < gradients[i].length; j++){
                this.nodes[i].weights[j] -= gradients[i][j] * this.learningRate;
            }
            this.nodes[i].bias -= biasGradients[i] * this.learningRate;
        }

        // randomly creating neurons
        if(Math.random() < this.createNeuronProbability) this.connectToNearby(this.addNeuron());
    }

    connectToNearby(neuron){
        const nearby = this.spHash.findNearby(neuron.x, neuron.y);

        // const maxNearbyDist = this.neuronConnectionRange * 2;

        // let dist;
        for(let i = 0; i < nearby.length; i++){
            if(nearby[i] === neuron) continue;
            // dist = Math.sqrt((nearby[i].x - neuron.x)**2 + (nearby[i].y - neuron.y)**2);

            // if(/*(Math.random()*0.7+0.3)*/0 * this.connectionRate < (1 - dist / maxNearbyDist) /*** 2*/){
                if(nearby[i].isInput){
                    neuron.addConnection(nearby[i]);
                    continue;
                } else if(nearby[i].isOutput){
                    nearby[i].addConnection(neuron);
                    continue;
                }

                if(Math.random() < this.doubleConnectionChance){
                    neuron.addConnection(nearby[i]);
                    nearby[i].addConnection(neuron);
                } else if(Math.random() < this.connectionChance) {
                    if(nearby[i].x < neuron.x){
                        neuron.addConnection(nearby[i]);
                    } else {
                        nearby[i].addConnection(neuron);                                 
                    }
                }

                // if(Math.random() < 0.5){
                //     // connect this node to other node
                //     neuron.addConnection(nearby[i]);
                // } else {
                //     // connect other node to this node
                //     nearby[i].addConnection(neuron);
                // }
            // }
        }
    }

    fit(inputs, outputs, epochs=5){
        for(let i = 0; i < epochs; i++){
            this.train(inputs, outputs);
        }
    }

    backPropogate(){

    }

    // uniformly random within a circle
    randomNeuronPosition(){
        return {
            x: (Math.random()*2-1) * this.coordinateRadius + this.coordinateMiddle,
            y: (Math.random()*2-1) * this.coordinateRadius + this.coordinateMiddle
        }
    }
}

function shortAngleDist(a0,a1) {
    const max = Math.PI*2;
    const da = (a1 - a0) % max;
    return 2*da % max - da;
}