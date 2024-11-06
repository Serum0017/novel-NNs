const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const physicalBrainSize = 1;

const numInputs = data.inputs[0].length;
const numOutputs = data.outputs[0].length;

// TODO: steps in order:
// 1. make outputs hold weight across iterations so that signals dont die out
// 2. either:
//      a) test enhancing traditional NNs by adding connections across layers (will make less robust?)
// or   b) try generating curves to initalize neurons in a pattern to approximate layer based approaches

const NN = new FreeformNN({
    physicalBrainSize,
    numInputs,
    numOutputs,
    steps: 3,
    learningRate: 0.00005,
    initNeuronsOnEdge: true,

    initialGridSize: .4,
    neuronConnectionRange: physicalBrainSize / 6,

    ...importTraditionalNN(),

    // /*format: [[x,y], [x,y], ...] where x^2+y^2 < 1*/
    // initialHiddenNeurons: [[0,-.5],[0,.5]],

    // /*format: [[nodeFromIndex, nodeToIndex], ...]*/
    // initialNetworkConnections: [[0,3],[1,3],[3,2],[0,4],[1,4],[4,2]]
    // ...pregenerateNN(/*[15, 10]*//*[10,8]*/[5,4]),

    createNeuronProbability: 0.1
});

let flag = false;
let counter = 0;
for(let i = 0; i < pretrainedNN.layers.length; i++){
    for(let j = 0; j < pretrainedNN.layers[i].length; j++){
        if(counter === 12){
            counter++;
        } else if(j === pretrainedNN.layers[i].length-1 && i === pretrainedNN.layers.length-1){
            counter = 12;
        }
        const params = pretrainedNN.layers[i][j].trainableParams;
        const bias = params.pop();
        NN.nodes[counter].weights = params;
        NN.nodes[counter].bias = bias; 
        counter++;
    }
}

function importTraditionalNN(other=pretrainedNN){
    const NN = other;
    const initialHiddenNeurons = [];

    function subdivide(width, numRegions, margin){
        const withoutMargin = width - margin * (numRegions+1);
        return withoutMargin / numRegions;
    }

    const margin = 0.03;
    const layerWidth = subdivide(2, NN.layers.length-1, margin);

    let x = margin;

    let layerIndexes = [];

    layerIndexes[0] = [0,1,2,3,4,5,6,7,8,9,10,11];

    for(let i = 0; i < other.layers.length; i++){
        const layerHeight = subdivide(2, NN.layers[i].length, margin);
        let y = margin;
        for(let j = 0; j < other.layers[i].length; j++){
            if(layerIndexes[i] === undefined) layerIndexes[i] = [];
            if(i !== 0 && i !== other.layers.length-1){
                layerIndexes[i].push(initialHiddenNeurons.length+13);
                initialHiddenNeurons.push([x-1,y-1 + layerHeight / 2]);
            }

            y += margin + layerHeight;
        }

        x += margin + layerWidth;
    }

    layerIndexes[layerIndexes.length-1] = [12];

    // connections
    const initialNetworkConnections = [];// TODO: get connections working
    for(let i = 0; i < layerIndexes.length-1; i++){
        const j = i + 1;

        for(let a = 0; a < layerIndexes[i].length; a++){
            for(let b = 0; b < layerIndexes[j].length; b++){
                initialNetworkConnections.push([layerIndexes[i][a], layerIndexes[j][b]]);
            }
        }
    }

    return {initialHiddenNeurons, initialNetworkConnections};
}

// function pregenerateNN(layers=[10,20,30]){
//     const l = layers.length;

//     const increment = 2 / (l + 1);

//     const initialHiddenNeurons = [];
//     const initialNetworkConnections = [];

//     const maxNeuronsInALayer = Math.max(...layers);

//     const inputIndexes = [];
//     const outputIndexes = [];
//     for(let i = 0; i < numInputs; i++){
//         inputIndexes.push(i);
//     }
//     for(let i = 0; i < numOutputs; i++){
//         outputIndexes.push(numInputs+i);
//     }
//     const paddingLen = inputIndexes.length + outputIndexes.length;

//     let previousIndexes = inputIndexes;
//     let nextIndexes = [];

//     for(let i = 0; i < l; i++){
//         const x = -1 + increment * (1 + i);
//         for(let j = 0; j < layers[i]; j++){
//             const y = interpolate(-.9, 0.9, (j - layers[i]/2 + maxNeuronsInALayer/2) / (maxNeuronsInALayer-1));
//             const neuronIndex = initialHiddenNeurons.length+paddingLen; 
//             initialHiddenNeurons.push([x,y]);
//             nextIndexes.push(neuronIndex);
//             for(let k = 0; k < previousIndexes.length; k++){
//                 initialNetworkConnections.push([previousIndexes[k], neuronIndex]);
//             }
//         }

//         previousIndexes = nextIndexes;
//         nextIndexes = [];
//     }

//     // connect to outputs
//     for(let i = 0; i < previousIndexes.length; i++){
//         for(let j = 0; j < outputIndexes.length; j++){
//             initialNetworkConnections.push([previousIndexes[i], outputIndexes[j]]);
//         }
//     }

//     return {
//         initialHiddenNeurons,
//         initialNetworkConnections
//     }
// }

// physicalBrainSize,
// neuronConnectionRange=physicalBrainSize/6,
// numInputs,
// numOutputs,
// steps=10,
// learningRate=0.01,
// createNeuronProbability=0.0008,
// // connectionRate=1,

// initialGridSize=.3,

// // takes up .8 of the entire grid
// initialSpread=0.8,

// gridRandomOffsetMagnitude=.01,

// doubleConnectionChance=0.1,

let camera;
function initCamera(){
    camera = {
        x: canvas.w / 2,
        y: canvas.h / 2,
        xv: 0, yv: 0, sv: 0,
        zoom: 500, targetZoom: 600
    };
}

const TAU = Math.PI * 2;
function renderNetwork(){
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    camera.x -= camera.xv * 5;
    camera.y -= camera.yv * 5;
    handleScroll({deltaY: camera.sv * 60});
    camera.zoom = interpolate(camera.zoom, camera.targetZoom, 0.1);

    ctx.save();

    ctx.translate(camera.x, camera.y);

    ctx.translate(physicalBrainSize/2, physicalBrainSize/2);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-physicalBrainSize/2, -physicalBrainSize/2);

    // border
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 4.5/1920;
    ctx.beginPath();
    ctx.roundRect(0, 0, physicalBrainSize, physicalBrainSize, 100/1920);
    ctx.stroke();
    ctx.closePath();
    
    ctx.fillStyle = 'white';
    for(let i = 0; i < NN.nodes.length; i++){
        const n = NN.nodes[i];

        ctx.lineWidth = 1;

        ctx.fillStyle = 'white';
        ctx.font = '400 .01px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        // draw current node
        const nodePos = [n.x, n.y];
        ctx.beginPath();
        ctx.arc(...nodePos, 0.01, 0, TAU);
        ctx.fill();
        ctx.closePath();

        ctx.fillText(n.output.toFixed(2), nodePos[0], nodePos[1] - 22/1920);

        // draw connections
        const connections = n.connections;
        for(let j = 0; j < connections.length; j++){
            ctx.lineWidth = Math.max(1, Math.abs(n.weights[j]) / 100000) / 1920;
            const nextNodePos = [connections[j].x, connections[j].y];
            ctx.beginPath();
            ctx.moveTo(...nodePos);
            ctx.lineTo(...nextNodePos);
            ctx.stroke();
            ctx.closePath();

            drawArrowOnLine(nodePos, nextNodePos);
        }
    }

    ctx.lineWidth = 3;
    ctx.font = '400 56px Inter';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    let MSE = new Array(data.outputs[0].length).fill(0);
    for(let i = 0; i < data.inputs.length; i++){
        const output = NN.forward(data.inputs[i]);
        for(let j = 0; j < output.length; j++){
            MSE[j] += (output[j] - data.outputs[i][j]) ** 2;
        }
    }

    const t = "Error: [" + MSE.map(m => m.toFixed(4)).join(',') + "]";
    ctx.font = '500 1px Inter';
    const metrics = ctx.measureText(t);
    const width = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight;

    ctx.font = `500 ${physicalBrainSize / (width + 0)}px Inter`;

    ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);


    // XOR Error vis

    // let outputs = new Array(data.inputs.length).fill(0);
    // for(let i = 0; i < outputs.length; i++){
    //     outputs[i] = NN.forward(data.inputs[i]);
    // }

    // const t = "outputs: [" + outputs.map(arr => arr.map(m => m.toFixed(4))).join(',') + "]";

    // ctx.font = '500 1px Inter';
    // const width = ctx.measureText(t).width;

    // ctx.font = `500 ${physicalBrainSize / (width + 0)}px Inter`;

    // ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);

    ctx.restore();
}

window.addEventListener("keydown", (e)=>{return handleKey(e, true)});
window.addEventListener("keyup", (e)=>{return handleKey(e, false)});
window.addEventListener("wheel", (e)=>{return handleScroll(e);});

let input = {
    up: false,
    down: false,
    left: false,
    right: false,
    shift: false,
    z: false,
    x: false
};
function handleKey(e, isDown) {
    if(e.repeat) return e.preventDefault();

    if(e.code === 'Digit0' && isDown){
        return initCamera();
    } else if(e.code === 'KeyZ' || e.code === 'KeyX'){
        if(e.code === 'KeyZ') input.z = isDown;
        else input.x = isDown;
        camera.sv = input.x - input.z;
    }

    if(e.code === 'KeyW'){
        input.up = isDown;
    } else if(e.code === 'KeyA'){
        input.right = isDown;
    } else if(e.code === 'KeyS'){
        input.down = isDown;
    } else if(e.code === 'KeyD'){
        input.left = isDown;
    } else if(e.code === 'ShiftLeft' || e.code === 'ShiftRight'){
        input.shift = isDown;
    }

    camera.xv = (input.left - input.right);
    camera.yv = (input.down - input.up);

    if(input.shift === true) {
        camera.xv *= 3;
        camera.yv *= 3;
    }
}
function handleScroll(e){
    camera.targetZoom *= 1 - e.deltaY * 0.6 / 1000;
}

function interpolate(start, end, t){
    return (1-t) * start + end * t;
}

const width = 1600;
const height = 900;
function resize() {
    canvas.w = canvas.width = window.innerWidth;
    canvas.h = canvas.height = window.innerHeight;
    ctx.lineCap = 'round';

    initCamera();
}
resize();
window.addEventListener('resize', resize);

const arrowAngle = Math.PI / 4;
const arrowLength = 12/1920;
function drawArrowOnLine([startX, startY], [endX, endY]){
    const middle = [
        (startX + endX) / 2,
        (startY + endY) / 2,
    ]

    const angle = Math.atan2(endY - startY, endX - startX);

    ctx.beginPath();
    ctx.translate(...middle);
    ctx.rotate(angle + arrowAngle / 2);
    ctx.moveTo(0,0);
    ctx.lineTo(arrowLength, 0);
    ctx.moveTo(0,0);
    ctx.rotate(-arrowAngle);
    ctx.lineTo(arrowLength, 0);
    ctx.moveTo(0,0);
    ctx.rotate(-angle + arrowAngle / 2);
    ctx.stroke();
    ctx.translate(-middle[0], -middle[1]);
    ctx.closePath();
}

function text(txt, x,y){
    ctx.strokeText(txt,x,y);
    ctx.fillText(txt,x,y);
}

function interpolate(a,b,t){
    return (1-t) * a + b * t;
}

function run(){
    NN.train(data.inputs, data.outputs, /*(progressPercent) => {console.log(`${Math.round(progressPercent*100)}% completed!`)}*/);

    renderNetwork();

    // requestAnimationFrame(run);

    setTimeout(() => {
        run();
    }, 100)
}

run();

// const firstLastNeurons = [
//     NN.addZeroNeuron({x:0.5,y:0.5+0.1}),
//     NN.addZeroNeuron({x:0.5,y:0.5-0.1}),
// ]

// for(let i = 0; i < firstLastNeurons.length; i++){
//     for(let j = 0; j < NN.numInputs; j++){
//         firstLastNeurons[i].addConnection(NN.nodes[j]);
//     }
//     NN.nodes[NN.numInputs].addConnection(firstLastNeurons[i]);
// }

// for(let i = 0; i < firstLastNeurons.length; i++){
//     firstLastNeurons[i].weights.fill(0);
// }

// const firstThirdNeurons = [
//     NN.addZeroNeuron({x:0.45,y:0.5+0.07}),
//     NN.addZeroNeuron({x:0.45,y:0.5-0.07}),
// ]

// for(let i = 0; i < firstThirdNeurons.length; i++){
//     // first layer
//     for(let j = 0; j < NN.numInputs; j++){
//         firstThirdNeurons[i].addConnection(NN.nodes[j]);
//     }

//     // third layer
//     for(let j = 23; j < 31; j++){
//         NN.nodes[j].addConnection(firstThirdNeurons[i]);
//     }
// }

// for(let i = 0; i < firstThirdNeurons.length; i++){
//     firstThirdNeurons[i].weights.fill(0);
// }

// const secondLastNeurons = [
//     NN.addZeroNeuron({x:0.55,y:0.5+0.07}),
//     NN.addZeroNeuron({x:0.55,y:0.5-0.07}),
// ]

// for(let i = 0; i < secondLastNeurons.length; i++){
//     // second layer
//     for(let j = 13; j < 23; j++){
//         secondLastNeurons[i].addConnection(NN.nodes[j]);
//     }

//     // last layer
//     NN.nodes[NN.numInputs].addConnection(secondLastNeurons[i]);
// }

// for(let i = 0; i < secondLastNeurons.length; i++){
//     secondLastNeurons[i].weights.fill(0);
// }



const firstNeurons = [
    NN.addZeroNeuron({x:0.345,y:0.5+0.1}),
    NN.addZeroNeuron({x:0.345,y:0.5}),
    NN.addZeroNeuron({x:0.345,y:0.5-0.1}),
]

for(let i = 0; i < firstNeurons.length; i++){
    // first layer
    for(let j = 0; j < NN.numInputs; j++){
        firstNeurons[i].addConnection(NN.nodes[j]);
    }

    // third layer
    for(let j = 23; j < 31; j++){
        NN.nodes[j].addConnection(firstNeurons[i]);
    }
}

for(let i = 0; i < firstNeurons.length; i++){
    firstNeurons[i].weights.fill(0);
}

const secondNeurons = [
    NN.addZeroNeuron({x:0.565,y:0.5+0.1}),
    NN.addZeroNeuron({x:0.565,y:0.5}),
    NN.addZeroNeuron({x:0.565,y:0.5-0.1}),
]

for(let i = 0; i < secondNeurons.length; i++){
    // second layer
    for(let j = 13; j < 23; j++){
        secondNeurons[i].addConnection(NN.nodes[j]);
    }

    // output
    NN.nodes[NN.numInputs].addConnection(secondNeurons[i]);   
}

for(let i = 0; i < secondNeurons.length; i++){
    secondNeurons[i].weights.fill(0);
}