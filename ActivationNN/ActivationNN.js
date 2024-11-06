class TraditionalNN {
    constructor({
        learningRate=0.01,
        NeuronClass=Neuron,
        layerSizes=[10,10,10],
        coordinateDescentTime=Infinity
    }){
        this.learningRate = learningRate;

        this.layers = [];

        for(let i = 0; i < layerSizes.length; i++){
            this.layers[i] = [];
            for(let j = 0; j < layerSizes[i]; j++){
                this.layers[i][j] = new NeuronClass(layerSizes[i-1] ?? 0);// numWeights
            }
        }

        this.nodeOutputs = [];
        this.weightedInputs = [];

        // this.gradients = [];
        // for(let i = 0; i < this.layers.length; i++){
        //     this.gradients[i] = [];
        //     for(let j = 0; j < this.layers[i].length; j++){
        //         this.gradients[i][j] = [];
        //         for(let k = 0; k < this.layers[i][j].trainableParams.length; k++){
        //             this.layers[i][j][k] = 0;
        //         }
        //     }
        // }

        // this.trainingWandBs = false;
        // this.coordinateCounter = 0;
        // this.coordinateDescentTime = coordinateDescentTime;
    }

    MSE(arr1, arr2){
        let sum = 0;
        for(let i = 0; i < arr1.length; i++){
            sum += (arr1[i] - arr2[i]) ** 2;
        }
        return sum;
    }

    calculateError(inputs=[[]], outputs=[[]]){
        let error = 0;
        for(let i = 0; i < inputs.length; i++){
            const randomInd = Math.random(Math.floor(inputs.length));
            error += this.MSE(this.forward(inputs[randomInd]), outputs[randomInd]);
        }
        return error;
    }

    calculateRealError(inputs=[[]], outputs=[[]]){
        let error = 0;
        for(let i = 0; i < inputs.length; i++){
            error += this.MSE(this.forward(inputs[i]), outputs[i]);
        }
        return error;
    }

    forward(input=[]){
        let curInput = input;
        let nextInput = [];

        // keep track of nodeOutputs (after activation function) and weightedInputs (before activation function)
        this.nodeOutputs = [];
        this.nodeOutputs.push(input);

        this.weightedInputs = new Array(this.layers.length).fill([]);
        this.weightedInputs[0] = input;

        for(let i = 1; i < this.layers.length; i++){
            for(let j = 0; j < this.layers[i].length; j++){
                const weightedInput = this.layers[i][j].sumWeighted(curInput);
                this.weightedInputs[i][j] = weightedInput;

                nextInput[j] = this.layers[i][j].process(curInput, weightedInput);
            }

            curInput = nextInput;
            this.nodeOutputs.push(nextInput);
            nextInput = [];
        }

        return curInput;
    }

    train(inputs=[[]], outputs=[[]]){
        for(let i = 0; i < inputs.length; i++){
            const randomIndex = Math.floor(Math.random() * inputs.length);
            this.updateGradients(inputs[randomIndex], outputs[randomIndex]);
        }

        // if(this.trainingWandBs === false){
        //     this.coordinateCounter++;
        //     if(this.coordinateCounter === this.coordinateDescentTime) {
        //         this.trainingWandBs = true;
        //     }
        // }
        
        // this.trainSlow(inputs, outputs);
    }

    // This is the central function. It applies gradients to the layer using backprop and only requires running an input through the network once.
    updateGradients(input=[], targetOutput=[]){
        // before we can do backprop, we have to do the last layer. We start by running through the network
        const output = this.forward(input);

        // and then we find the nodeValues (useful data for calculating the partial derivatives)
        let nodeValues = this.calculateOutputNodeValues(targetOutput, output);
        this.updateGradientsOfLayer(nodeValues, this.layers.length-1);

        // backprop!
        for(let currentLayer = this.layers.length - 2; currentLayer >= 1; currentLayer--){
            nodeValues = this.calculateHidddenLayerNodeValues(currentLayer + 1, nodeValues);
            this.updateGradientsOfLayer(nodeValues, currentLayer);
            // use those nodevalues to update the gradients of the hidden layer
        }
    }

    updateGradientsOfLayer(nodeValues=[], layerInd){// layerInd >= 1
        // goal is to calculate a derivative for each weight
        const layer = this.layers[layerInd];
        const lastLayerLen = this.layers[layerInd-1].length;
        const lastNodeOutputs = this.nodeOutputs[layerInd-1];

        // let gradients = new Array(layer.length).fill(new Array(lastLayerLen).fill(0));
        // let biasGradients = new Array(layer.length).fill(0);
        for(let i = 0; i < layer.length; i++){

            if(window.trainingMode === 'traditional'/*this.trainingWandBs === true*/ /*|| window.trainingTraditional*/){
                for(let j = 0; j < lastLayerLen; j++){
                    const derivative = lastNodeOutputs[j] * nodeValues[i];
                    // gradients[i][j] += derivative;
                    layer[i].trainableParams[j] += derivative * this.learningRate;
                }
    
                // bias
                const biasDerivative = 1 * nodeValues[i];
                // biasGradients[i] += biasDerivative;
    
                layer[i].trainableParams[lastLayerLen]/*bias*/ += biasDerivative * this.learningRate;
            } else {
                for(let j = lastLayerLen+1; j < layer[i].trainableParams.length; j++){
                    // // accurate
                    // const activationfnWRTparam = layer[i].activationParamDerivative(lastNodeOutputs, this.weightedInputs[layerInd][i]);
                    // const goodPart = nodeValues[i] / layer[i].processDerivative(lastNodeOutputs, this.weightedInputs[layerInd][i]);
                    // const extraParamDeriv = activationfnWRTparam * goodPart;

                    // if(Number.isFinite(extraParamDeriv) === false) continue;

                    // layer[i].trainableParams[j] += extraParamDeriv * this.learningRate * CONSTANTS.extraParamLearningCoefficients[j - lastLayerLen - 1];


                    // // faulty but better performance.
                    const extraParamDeriv = nodeValues[i] / layer[i].activationParamDerivative(this.nodeOutputs[layerInd-1], this.weightedInputs[layerInd][i]);// / processderiv
                    if(Number.isFinite(extraParamDeriv) === false) continue;
                    layer[i].trainableParams[j] -= extraParamDeriv * this.learningRate; // this is 1 -> * CONSTANTS.extraParamLearningCoefficients[j - lastLayerLen - 1];


    
                    // if(CONSTANTS.extraParamLimits[j - lastLayerLen - 1] !== undefined){
                    //     layer[i].trainableParams[j] = Math.max(layer[i].trainableParams[j], CONSTANTS.extraParamLimits[j - lastLayerLen - 1][0]);
                    //     layer[i].trainableParams[j] = Math.min(layer[i].trainableParams[j], CONSTANTS.extraParamLimits[j - lastLayerLen - 1][1]);
                    // }
                }
            }
        }
    }

    calculateHidddenLayerNodeValues(outputLayerIndex, oldNodeValues=[]){// outputLayerIndex >= 2
        // we have an output layer and input layer. This function calculates the new nodeValues for the input layer given the output layer's.
        const numNodesInLayer = this.layers[outputLayerIndex - 1].length;
        let nodeValues = new Array(numNodesInLayer);

        // i = input layer index
        for(let i = 0; i < nodeValues.length; i++){
            let nodeValue = 0;
            // j = output layer index

            // sum of for each connection to node j (for all j), weights connecting the ith node to the jth * the old node value of j
            for(let j = 0; j < oldNodeValues.length; j++){
                let weightedInputDerivative = this.layers[outputLayerIndex][j].trainableParams[i] * oldNodeValues[j];
                nodeValue += weightedInputDerivative;
            }
            // outputLayerIndex-2 = the layer before the input.

            // multiply by the derivative of the activation function with respect to its weighted input
            nodeValue *= this.layers[outputLayerIndex-1][i].processDerivative(this.nodeOutputs[outputLayerIndex-2], this.weightedInputs[outputLayerIndex-1][i]);
            nodeValues[i] = nodeValue;
        }
        return nodeValues;
    }

    // for a single data point, find a useful product of derivatives. This is only for the output layer.
    // nodeValues is a bad name for a variable that's some of the partial derivatives that are useful to compute doutput / dweight or doutput / dbias eventually.
    calculateOutputNodeValues(trueOutput=[], calculatedOutput=[]/*output already calculated by the network for a single data point*/) {
        let nodeValues = new Array(calculatedOutput.length);

        for(let i = 0; i < nodeValues.length; i++){
            const costDerivative = this.MSEDerivative(trueOutput, calculatedOutput);
            const activationDerivative = this.layers[this.layers.length-1][i].processDerivative(this.nodeOutputs[this.layers.length-2], this.weightedInputs[this.layers.length-1][i]);
            nodeValues[i] = costDerivative * activationDerivative;
        }

        return nodeValues;
    }

    
    MSEDerivative(arr1, arr2){
        let sum = 0;
        for(let i = 0; i < arr1.length; i++){
            sum += /*2 **/ (arr1[i] - arr2[i]); // not including the 2 from the derivative for speed. Instead, use a higher learning rate.
        }
        return sum;
    }

    // trainSlow(inputs=[[]], outputs=[[]]){
    //     let error = this.calculateError(inputs, outputs);

    //     for(let i = 0; i < this.layers.length; i++){
    //         for(let j = 0; j < this.layers[i].length; j++){
    //             for(let k = 0; k < this.layers[i][j].trainableParams.length; k++){
    //                 this.layers[i][j].trainableParams[k] += this.h;
    //                 this.gradients[i][j][k] = (this.calculateError(inputs, outputs) - error);
    //                 this.layers[i][j].trainableParams[k] -= this.h;
    //             }
    //         }
    //     }

    //     // console.log(error, this.gradients);

    //     for(let i = 0; i < this.layers.length; i++){
    //         for(let j = 0; j < this.layers[i].length; j++){
    //             for(let k = 0; k < this.layers[i][j].trainableParams.length; k++){
    //                 this.layers[i][j].trainableParams[k] += this.learningRate * this.gradients[i][j][k];
    //             }
    //         }
    //     }
    // }
}