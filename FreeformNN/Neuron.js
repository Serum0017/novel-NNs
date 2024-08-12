let globalId = -1;
globalThis.generateId = () => {
    return ++globalId;
}
let sum;
class Neuron {
    constructor({x,y}){
        this.id = generateId();
        this.x = x;
        this.y = y;
        this.input = 0;// input strength
        this.output = -1;

        // incoming weights
        this.weights = [];
        this.bias = 0;

        // reference to incoming connections
        this.connections = [];
    }
    calculateOutput(){
        const inputs = this.connections.map(c => c.output);
        sum = 0;
        for(let i = 0; i < inputs.length; i++){
            sum += inputs[i] * this.weights[i];
        }
        // we divide by length because when new connections are added, we on average want the same output
        return this.activationFunction(sum / inputs.length + this.bias);
    }
    sigmoid(x){
        return 1 / (1 + Math.exp(-x));
    }
    // tanh
    activationFunction(x){
        return 2 * this.sigmoid(2*x) - 1;
    }
}