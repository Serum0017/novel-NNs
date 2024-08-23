const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const width = 1600;
const height = 900;
let screenBounds;
function resize(){
    window.resizeElements([canvas]);
    canvas.w = canvas.width;
    canvas.h = canvas.height;
    recalcScreenBounds();
}

window.resizeElements = function (elements) {
    for (const element of elements) {
        if (element.width !== width) {
            element.width = width;
            element.style.width = `${width}px`;
        }
        if (element.height !== height) {
            element.height = height;
            element.style.height = `${height}px`;
        }
        let scaleMult = element?._scaleMult ?? 1;
        element.style.transform = `scale(${
            Math.min(window.innerWidth / width, window.innerHeight / height) *
            scaleMult
        })`;
        element.style.left = `${(window.innerWidth - width) / 2}px`;
        element.style.top = `${(window.innerHeight - height) / 2}px`;
    }
    return Math.min(window.innerWidth / width, window.innerHeight / height);
};
  
window.addEventListener('resize', function () {
    resize();
});
resize();

const physicalBrainSize = 1;

const NN = new FreeformNN({
    physicalBrainSize,
    numInputs: 2,
    numOutputs: 1,
    steps: 5,
    learningRate: 0.01
});

const TAU = Math.PI * 2;
function renderNetwork(){
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,canvas.w,canvas.h);

    ctx.globalAlpha = 0.3;
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.w,canvas.h);

    ctx.globalAlpha = 1;
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(canvas.w / 2, canvas.h / 2, canvas.h * 0.4, 0, TAU);
    ctx.fill();
    ctx.closePath();

    ctx.strokeStyle = 'black';
    for(let i = 0; i < NN.nodes.length; i++){
        const n = NN.nodes[i];

        ctx.lineWidth = 1;

        // draw current node
        const nodePos = nodeToWorld(n.x, n.y);
        ctx.beginPath();
        ctx.arc(...nodePos, 10, 0, TAU);
        ctx.stroke();
        ctx.closePath();

        // draw connections
        const connections = n.connections;
        for(let j = 0; j < connections.length; j++){
            ctx.lineWidth = Math.max(1, Math.abs(n.weights[j]));
            const nextNodePos = nodeToWorld(connections[j].x, connections[j].y);
            ctx.beginPath();
            ctx.moveTo(...nodePos);
            ctx.lineTo(...nextNodePos);
            ctx.stroke();
            ctx.closePath();
        }
    }

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
    ctx.font = '400 56px Grandstander';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 3;

    let MSE = new Array(xor.inputs[0].length).fill(0);
    for(let i = 0; i < xor.inputs.length; i++){
        const outputs = NN.forward(xor.inputs[i]);
        for(let j = 0; j < outputs.length; j++){
            MSE[j] += (outputs[j] - xor.outputs[i][j]) ** 2;
        }
    }
    text("MSE: [" + MSE.map(m => m.toFixed(4)).join(',') + "]", canvas.w / 2, canvas.h * 0.05);
}

function text(txt, x,y){
    ctx.strokeText(txt,x,y);
    ctx.fillText(txt,x,y);
}

function recalcScreenBounds(){
    screenBounds = {
        x: {
            top: canvas.w / 2 - canvas.h * 0.4,
            bottom: canvas.w / 2 + canvas.h * 0.4
        },
        y: {
            top: canvas.h / 2 - canvas.h * 0.4,
            bottom: canvas.h / 2 + canvas.h * 0.4
        },
    }
}
function nodeToWorld(x,y){
    return [
        interpolate(screenBounds.x.top, screenBounds.x.bottom, x / physicalBrainSize),
        interpolate(screenBounds.y.top, screenBounds.y.bottom, y / physicalBrainSize),
    ]
}

function interpolate(a,b,t){
    return (1-t) * a + b * t;
}

const xor = {
    inputs: [
        [0, 0]
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

function run(){
    NN.train(xor.inputs, xor.outputs);

    renderNetwork();

    // requestAnimationFrame(run);

    setTimeout(() => {
        run();
    }, 100)
}

run();