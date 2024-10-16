/*let*/const CONSTANTS = Object.freeze({
    // extraParams: 0,
    // activationFunction: 'LeakyReLU',
    // extraParamLearningCoefficients: [],
    // extraParamLimits: []

    extraParams: 1,
    activationFunction: /*'LeakyReLU',*/'DynamicLeakyReLU',//'sigmoid',//'DynamicLeakyReLU',//,//'classic',
    extraParamLearningCoefficients: [1],//[1 / 100],
    extraParamLimits: [/*[0.001, 0.1]*/]
})

// window.redefineConstants = () => {
//     if(window.trainingTraditional){
//         CONSTANTS = Object.freeze({
//             extraParams: 0,
//             activationFunction: 'LeakyReLU',
//             extraParamLearningCoefficients: [],
//             extraParamLimits: []
//         })
//     } else {
//         CONSTANTS = Object.freeze({
//             extraParams: 1,
//             activationFunction: /*'LeakyReLU',*/'DynamicLeakyReLU',//'sigmoid',//'DynamicLeakyReLU',//,//'classic',
//             extraParamLearningCoefficients: [1],//[1 / 100],
//             extraParamLimits: [/*[0.001, 0.1]*/]
//         })
//     }
// }

let sum, trainableParamsLen;
class Neuron {
    constructor(numWeights){
        trainableParamsLen = CONSTANTS.extraParams + numWeights + 1/*bias*/;
        this.trainableParams = new Array(trainableParamsLen);
        for(let i = 0; i < this.trainableParams.length; i++) { this.trainableParams[i] = Math.random() * 2 - 1; }
        // instead of having learning rates, just divide in the activation function itself.
        // If you want a param to grow 10x as slowly, do param * 0.1 in the activ. fn.
        // this.trainableLearningRates = new Array(trainableParamsLen).fill(1);
        this.le = this.trainableParams.length;
        this.process = this[CONSTANTS.activationFunction];
        this.processDerivative = this[CONSTANTS.activationFunction + 'Derivative'];
        this.activationParamDerivative = this[CONSTANTS.activationFunction + 'DerivativeWRTActivationParam']// this will be a matrix in the general implementation but i'm only using a single param. For future work!

        // TEMP, only for DynamicLeakyReLU
        this.trainableParams[this.le-1] = 0.01;
    }
    sumWeighted(inputs=[]){
        sum = 0;
        for(let i = 0; i < inputs.length; i++){
            sum += inputs[i] * this.trainableParams[i];
        }
        sum += this.trainableParams[inputs.length];
        return sum;
    }
    // cutting edge, yet classic
    classic(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return Math.max(0, weightedInput);
    }
    classicDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return weightedInput > 0 ? 1 : 0;
    }

    LeakyReLU(inputs=[], weightedInput=this.sumWeighted(inputs)){
        if(weightedInput > 0) return weightedInput;
        else return .01 * weightedInput;
    }
    LeakyReLUDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return weightedInput > 0 ? 1 : .01;
    }

    classicSigmoid(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return 1 / (1 + Math.exp(-weightedInput));
    }
    classicSigmoidDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        let sig = this.sigmoid(inputs, weightedInput);
        return sig * (1 - sig);
    }

    sigmoid(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return 1 / (1 + Math.exp(-weightedInput*this.trainableParams[this.le-1]));
    }
    sigmoidDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        let sig = this.sigmoid(inputs, weightedInput);
        return sig * (1 - sig) * this.trainableParams[this.le-1];
    }

    ReLU(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return Math.max(this.trainableParams[this.le-1], weightedInput);
    }
    ReLUDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return weightedInput > this.trainableParams[this.le-1] ? 1 : 0;
    }

    DynamicLeakyReLU(inputs=[], weightedInput=this.sumWeighted(inputs)){
        if(weightedInput > 0) return weightedInput;
        else return this.trainableParams[this.le-1] * weightedInput;
    }
    DynamicLeakyReLUDerivative(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return weightedInput > 0 ? 1 : this.trainableParams[this.le-1];
    }
    // derivative with respect to the parameter
    DynamicLeakyReLUDerivativeWRTActivationParam(inputs=[], weightedInput=this.sumWeighted(inputs)){
        return weightedInput > 0 ? 0 : weightedInput;
    }
}