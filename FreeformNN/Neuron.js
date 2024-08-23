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
        this.output = 0;

        // output/ input node booleans
        this.isInput = false;
        this.isOutput = false;

        // incoming weights
        this.weights = [];
        this.bias = 0;

        // reference to incoming connections
        this.connections = [];
    }
    calculateOutput(){
        const inputs = this.connections.map(c => c.output);
        if(inputs.length === 0) {
            if(this.isInput) return this.output; 
            else return this.activationFunction(this.bias);
        }
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
    ReLU(x){
        return Math.max(0, x);
    }

    addConnection(other){
        this.connections.push(other);
        this.weights.push(Math.random()-0.5);
    }
}