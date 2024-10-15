class DescisionTree {
    constructor(depth, numChoices){
        this.layers = [];

        for(let i = 0; i < depth; i++){
            this.layers[i] = [];
            for(let j = 0; j < numChoices ** i; j++){
                this.layers[i].push(new DescisionStump(numChoices));
            }
        }

        this.numChoices = numChoices;
    }
    decide(input){
        return this._decide(0, 0, input);
    }
    _decide(index, layer, input){
        // return this.layers[0][0].decide(input)[0];
        if(layer === this.layers.length) return input;
        const [output, outputBranch] = this.layers[layer][index].decide(input);
        return this._decide(index*this.numChoices+outputBranch, layer+1, output);
    }
}

class DescisionStump {
    constructor(numChoices){
        this.thresholds = new Array(numChoices-1);
        this.outputs = new Array(numChoices);

        this.bias = Math.random() * 2 - 1;

        for(let i = 0; i < numChoices-1; i++){
            this.thresholds[i] = Math.random() * 2 - 1;
            this.outputs[i] = Math.random() * 2 - 1;
        }
        this.outputs[numChoices-1] = Math.random() * 2 - 1;

        this.thresholds.sort();
    }
    test(){
        let sum = 0;
        for(let i = 0; i < this.thresholds.length; i++){
            sum += this.thresholds[i];
        }
        for(let i = 0; i < this.outputs.length; i++){
            sum += this.outputs[i];
        }
        return [sum, 0];
    }
    decide(input){
        // return [this.test()[0], 0];
        // return [(this.outputs[0] + this.thresholds[0]), 0];
        if(input < this.thresholds[0]) return [this.interpolate(this.outputs[0], this.outputs[1], (input - this.thresholds[0]) / (this.thresholds[1] - this.thresholds[0])) + this.bias, 0];
        for(let i = 1; i < this.thresholds.length; i++){
            if(input < this.thresholds[i]){
                return [this.interpolate(this.outputs[i-1], this.outputs[i], (input - this.thresholds[i-1]) / (this.thresholds[i] - this.thresholds[i-1])) + this.bias, i-1];
            }
        }
        return [this.interpolate(this.outputs[this.outputs.length-2], this.outputs[this.outputs.length-1], (input - this.thresholds[this.thresholds.length-2]) / (this.thresholds[this.thresholds.length-1] - this.thresholds[this.thresholds.length-2])) + this.bias, this.outputs.length-1];
    }
    interpolate(start, end, t){
        return (1-t) * start + t * end;
    }
    // interpolateActivationFunction(t){
    //     return t;
    // }
}