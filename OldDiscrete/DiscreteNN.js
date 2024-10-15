class DiscreteNN {
    // hyperparameters:
    constructor({
        learningRate=0.01,
        layerSizes=[10,10,10],
        dtreedepth=1,
        numChoices=3
    }){
        this.learningRate = learningRate;
        this.h = 0.0001;

        this.layers = [];
        for(let i = 0; i < layerSizes.length; i++){
            const layerArr = [];
            for(let j = 0; j < layerSizes[i]; j++){
                layerArr.push(new DescisionTree(dtreedepth, numChoices));
            }
            this.layers.push(layerArr);
        }

        this.gradientArr = [];

        for(let i = 0; i < this.layers.length; i++){
            this.gradientArr[i] = [];
            for(let j = 0; j < this.layers[i].length; j++){
                this.gradientArr[i][j] = [];
                const tree = this.layers[i][j];
                for(let k = 0; k < tree.layers.length; k++){
                    this.gradientArr[i][j][k] = [];
                    for(let l = 0; l < tree.layers[k].length; l++){
                        const stump = tree.layers[k][l];
                        this.gradientArr[i][j][k][l] = new Array(stump.thresholds.length + stump.outputs.length).fill(0);
                    }
                }
            }
        }
    }

    // single number. If this shows any promise then i can upgrade to taking in multiple numbers in a single input (dtree random selection)
    forward(input){
        let currentInput = input;
        let nextInput = 0;
        for(let i = 0; i < this.layers.length; i++){
            for(let j = 0; j < this.layers[i].length; j++){
                nextInput += this.layers[i][j].decide(currentInput);// maybe weight each dtree not per neuron, per layer?\
                // console.log(nextInput, {i, j});
            }
            currentInput = nextInput;
            nextInput = 0;
        }

        return currentInput;
    }

    calculateError(inputs=[], outputs=[]){
        let error = 0;
        for(let i = 0; i < inputs.length; i++){
            error += (this.forward(inputs[i]) - outputs[i]) ** 2;
        }
        return error;
    }

    // arrays of single numbers
    train(inputs=[], outputs=[]){
        let error = this.calculateError(inputs, outputs);

        for(let i = 0; i < this.layers.length; i++){
            for(let j = 0; j < this.layers[i].length; j++){
                const tree = this.layers[i][j];
                for(let k = 0; k < tree.layers.length; k++){
                    for(let l = 0; l < tree.layers[k].length; l++){
                        const stump = tree.layers[k][l];

                        for(let m = 0; m < stump.thresholds.length; m++){
                            stump.thresholds[m] += this.h;
                            const newError = this.calculateError(inputs, outputs);
                            stump.thresholds[m] -= this.h;

                            this.gradientArr[i][j][k][l][m] = (newError - error);
                        }

                        for(let m = 0; m < stump.outputs.length; m++){
                            stump.outputs[m] += this.h;
                            const newError = this.calculateError(inputs, outputs);
                            stump.outputs[m] -= this.h;

                            this.gradientArr[i][j][k][l][m+stump.thresholds.length] = (newError - error);
                        }

                        stump.bias += this.h;
                        const newError = this.calculateError(inputs, outputs);
                        stump.bias -= this.h;

                        this.gradientArr[i][j][k][l][stump.outputs.length+stump.thresholds.length] = (newError - error);
                    }
                }
            }
        }

        for(let i = 0; i < this.layers.length; i++){
            for(let j = 0; j < this.layers[i].length; j++){
                const tree = this.layers[i][j];
                for(let k = 0; k < tree.layers.length; k++){
                    for(let l = 0; l < tree.layers[k].length; l++){
                        const stump = tree.layers[k][l];

                        for(let m = 0; m < stump.thresholds.length; m++){
                            stump.thresholds[m] -= this.gradientArr[i][j][k][l][m] * this.learningRate;
                        }

                        for(let m = 0; m < stump.outputs.length; m++){
                            stump.outputs[m] -= this.gradientArr[i][j][k][l][m+stump.thresholds.length] * this.learningRate;
                        }

                        stump.bias -= this.gradientArr[i][j][k][l][stump.outputs.length+stump.thresholds.length] * this.learningRate;

                        stump.thresholds.sort();
                    }
                }
            }
        }
    }

    fit(inputs, outputs, epochs=5){
        for(let i = 0; i < epochs; i++){
            this.train(inputs, outputs);
        }
    }
}