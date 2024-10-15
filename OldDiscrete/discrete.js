const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const physicalBrainSize = 1;

const NN = new DiscreteNN({
    learningRate: 0.001,
    layerSizes: [3,2],
    dtreedepth: 3,
    numChoices: 3
});

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

    const margin = 0.03;
    const layerWidth = subdivide(1, NN.layers.length, margin);

    let x = margin;
    for(let i = 0; i < NN.layers.length; i++){
        // const layerStart = x;
        // const layerEnd = x + margin;
        const layerHeight = subdivide(1, NN.layers[i].length, margin);
        let y = margin;
        for(let j = 0; j < NN.layers[i].length; j++){
            drawDescisionTree(x, y, layerWidth, layerHeight, NN.layers[i][j]);
            y += margin + layerHeight;
        }
        x += margin + layerWidth;
    }

    // for(let i = 0; i < NN.nodes.length; i++){
    //     const n = NN.nodes[i];

    //     ctx.lineWidth = 1;

    //     ctx.fillStyle = 'white';
    //     ctx.font = '400 .01px Inter';
    //     ctx.textAlign = 'center';
    //     ctx.textBaseline = 'bottom';

    //     // draw current node
    //     const nodePos = [n.x, n.y];
    //     ctx.beginPath();
    //     ctx.arc(...nodePos, 0.01, 0, TAU);
    //     ctx.fill();
    //     ctx.closePath();

    //     ctx.fillText(n.output.toFixed(2), nodePos[0], nodePos[1] - 22/1920);

    //     // draw connections
    //     const connections = n.connections;
    //     for(let j = 0; j < connections.length; j++){
    //         ctx.lineWidth = Math.max(1, Math.abs(n.weights[j]) / 100000) / 1920;
    //         const nextNodePos = [connections[j].x, connections[j].y];
    //         ctx.beginPath();
    //         ctx.moveTo(...nodePos);
    //         ctx.lineTo(...nextNodePos);
    //         ctx.stroke();
    //         ctx.closePath();

    //         drawArrowOnLine(nodePos, nextNodePos);
    //     }
    // }

    ctx.lineWidth = 3;
    ctx.font = '400 56px Inter';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';


    const t = "Target: [" + data.outputs.toString() + "]" + " Outputs: [" + data.outputs.map(o => NN.forward(o).toFixed(2)).toString() + "]";
    ctx.font = '500 1px Inter';
    const metrics = ctx.measureText(t);
    const width = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight;

    ctx.font = `500 ${physicalBrainSize / (width + 0)}px Inter`;

    ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);

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

function drawDescisionTree(x,y,w,h,d){
    ctx.strokeRect(x,y,w,h);

    const margin = 0.01;
    const layerWidth = subdivide(w, d.layers.length, margin);

    let smolX = margin;
    for(let i = 0; i < d.layers.length; i++){
        // const layerStart = x;
        // const layerEnd = x + margin;
        const layerHeight = subdivide(h, d.layers[i].length, margin);
        let smolY = margin;
        for(let j = 0; j < d.layers[i].length; j++){
            drawDescisionStump(x + smolX, y + smolY, layerWidth, layerHeight, d.layers[i][j]);
            smolY += margin + layerHeight;
        }
        smolX += margin + layerWidth;
    }
}

function drawDescisionStump(x,y,w,h,s){
    // ctx.beginPath();
    // ctx.arc(x + w/2, y+h/2, Math.abs(Math.min(w,h)/2), 0, Math.PI * 2);
    // ctx.stroke();
    // ctx.closePath();
    ctx.setLineDash([0.01,0.01]);
    ctx.strokeRect(x,y,w,h);
    ctx.setLineDash([]);

    // console.log(s);

    // circles on top = thresholds
    // circles on bottom = outputs

    const leftCircleX = x + w * 1/3;
    const rightCircleX = x + w * 2/3;
    const margin = 0.001;
    let smolY = y + margin;
    let layerWidth = subdivide(h, s.thresholds.length + s.outputs.length, margin);

    ctx.textAlign = 'left';
    ctx.font = `400 ${Math.max(0.01, layerWidth / 5)}px Inter`;

    for(let i = 0; i < s.thresholds.length + s.outputs.length; i++){
        const drawX = i%2 === 1 ? leftCircleX : rightCircleX;
        const drawY = smolY + layerWidth / 2
        ctx.beginPath();
        ctx.arc(drawX, drawY, layerWidth / 2 / 3 + margin, 0, Math.PI * 2);
        ctx.stroke();
        ctx.closePath();

        let val;
        if(i % 2 === 0){
            // outputs
            val = s.outputs[i / 2];
        } else {
            // thresholds
            val = s.thresholds[(i-1) / 2];
        }

        const text = val.toFixed(2);
        const t = ctx.measureText(text);
        const h = t.actualBoundingBoxAscent / 100;
        ctx.fillText(text, drawX + layerWidth / 2 / 3 + margin * 3, drawY + h/3);
        
        smolY += margin + layerWidth;
    }

    // for(let i = 0; i < s.thresholds.length; i++){
    //     ctx.beginPath();
    //     ctx.arc(smolX + layerWidth / 2, topCircleY, layerWidth / 2 / 3, 0, Math.PI * 2);
    //     ctx.stroke();
    //     ctx.closePath();
    //     smolX += margin + layerWidth;
    // }

    // smolX = x + margin;
    // layerWidth = subdivide(w, s.outputs.length, margin);
    // for(let i = 0; i < s.outputs.length; i++){
    //     ctx.beginPath();
    //     ctx.arc(smolX + layerWidth / 2, bottomCircleY, layerWidth / 2 / 3, 0, Math.PI * 2);
    //     ctx.stroke();
    //     ctx.closePath();
    //     smolX += margin + layerWidth;
    // }
}

function subdivide(width, numRegions, margin){
    const withoutMargin = width - margin * (numRegions+1);
    return withoutMargin / numRegions;
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

data.inputs = data.inputs.map(d => d[0]);
data.outputs = data.outputs.map(d => d[0]);

function run(){
    NN.train(data.inputs, data.outputs);

    renderNetwork();

    requestAnimationFrame(run);

    // setTimeout(() => {
    //     run();
    // }, 100)
}

run();