// assume coordinate space 0 to totalHashDistance subdivided into binDistance
let x, y;
class SpatialHash {
    constructor(totalHashDistance, binDistance=totalHashDistance/10){
        this.totalHashDistance = totalHashDistance;
        this.binDistance = binDistance;
        this.positionsLen = this.totalHashDistance / this.binDistance;

        // positions: { x: {y: [entities at this hash] } }
        this.positions = [...Array(this.positionsLen)].map(_ => [...Array(this.positionsLen)]);
        for(x in this.positions){
            for(y in this.positions){
                this.positions[x][y] = [];
            }
        }
    }
    // entity must be within totalHashDistance
    // entity must be an object and must have a hashId
    addEntity(entity){
        x = Math.floor(entity.x / this.binDistance);
        y = Math.floor(entity.y / this.binDistance);

        // danger! this will error if we insert outside of the hash dist
        this.positions[x][y-1].push(entity);
        this.positions[x][y].push(entity);
        this.positions[x][y+1].push(entity);

        this.positions[x-1][y-1].push(entity);
        this.positions[x-1][y].push(entity);
        this.positions[x-1][y+1].push(entity);

        this.positions[x+1][y-1].push(entity);
        this.positions[x+1][y].push(entity);
        this.positions[x+1][y+1].push(entity);
    }
    removeEntity(entity){
        // entities should not move!
        x = Math.floor(entity.x / this.binDistance);
        y = Math.floor(entity.y / this.binDistance);

        for(let i = 0; i < this.positions[x][y].length; i++){
            if(this.positions[x][y] === entity){
                this.positions[x][y].splice(i,1);
                break;
            }
        }
    }
    findNearby(x,y){
        // For our use case, we assume that neurons are too far away if they're not in our spatial hash cell or any of the adjacent. We push to all of the adjacent spaces, so our lookup is fast.
        return this.positions[Math.floor(x / this.binDistance)][Math.floor(y / this.binDistance)];
    }
}
