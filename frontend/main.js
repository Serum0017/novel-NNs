const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const physicalBrainSize = 1;

const NN = new FreeformNN({
    physicalBrainSize,
    numInputs: 2,
    numOutputs: 1,
    steps: 8,
    learningRate: 0.3,
    initNeuronsOnEdge: true,

    initialGridSize: .4,
    neuronConnectionRange: physicalBrainSize / 6,

    /*format: [[x,y], [x,y], ...] where x^2+y^2 < 1*/
    initialHiddenNeurons: [[0,0]],

    /*format: [[nodeFromIndex, nodeToIndex], ...]*/
    initialNetworkConnections: [[0,3],[1,3],[3,2]]
});

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

    // let MSE = new Array(xor.outputs[0].length).fill(0);
    // for(let i = 0; i < xor.inputs.length; i++){
    //     const outputs = NN.forward(xor.inputs[i]);
    //     for(let j = 0; j < outputs.length; j++){
    //         MSE[j] += (outputs[j] - xor.outputs[i][j]) ** 2;
    //         // console.log(outputs[j], xor.outputs[i][j]);
    //     }
    // }

    // const t = "MSE: " + MSE.map(m => m.toFixed(4)).join(',');
    // ctx.font = '500 1px Inter';
    // const metrics = ctx.measureText(t);
    // const width = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight;

    // ctx.font = `500 ${physicalBrainSize / (width * 2)}px Inter`;

    // ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);

    let outputs = new Array(xor.inputs.length).fill(0);
    for(let i = 0; i < outputs.length; i++){
        outputs[i] = NN.forward(xor.inputs[i]);
    }

    const t = "outputs: [" + outputs.map(arr => arr.map(m => m.toFixed(4))).join(',') + "]";

    ctx.font = '500 1px Inter';
    const width = ctx.measureText(t).width;

    ctx.font = `500 ${physicalBrainSize / (width + 0)}px Inter`;

    ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);

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

const xor = {
    inputs: [
        [0,0],
        [0, 1],
        [1, 0],
        [1, 1]
    ],
    outputs: [
        [0],
        [1],
        [1],
        [0]
    ]
}

// TODO: Use new Dataset.

function run(){
    NN.train(xor.inputs, xor.outputs);

    renderNetwork();

    requestAnimationFrame(run);

    // setTimeout(() => {
    //     run();
    // }, 100)
}

run();