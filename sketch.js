let num = 8;
let w_grid,h_grid;
let grid_v = new Array(num);
let action_count = 0;

function setup() {
  createCanvas(500,500);
  frameRate(60);
  w_grid = width/num;
  h_grid = height/num;
  for (let i = 0; i < num; i++){
    grid_v[i] = new Array(num);
  }
  init_grid();
}

function draw() {
  background(51);
  //action();
  draw_grid();
  judgment();
}





function action(){
  let index_x = floor(random(num-0.001));
  let index_y = floor(random(num-0.001));
  turn(index_x, index_y);
  action_count += 1;
  console.log(action_count);
}

function mousePressed(){
  let index_x = floor(mouseX/w_grid);
  let index_y = floor(mouseY/w_grid);
  turn(index_x, index_y);
}


function turn(x, y){
  grid_v[x][y] = !grid_v[x][y];
  if(x+1 < num){ grid_v[x+1][y] = !grid_v[x+1][y] };
  if(x-1 >= 0){ grid_v[x-1][y] = !grid_v[x-1][y] };
  if(y+1 < num){ grid_v[x][y+1] = !grid_v[x][y+1] };
  if(y-1 >= 0){ grid_v[x][y-1] = !grid_v[x][y-1] };
}

function init_grid(){
  for (let i = 0; i < num; i++){
    for(let j = 0; j < num; j++){
      if(random(1)>0.5){
        grid_v[i][j] = true;
      } else {
        grid_v[i][j] = false;
      }
    }
  }
}

function judgment(){
  let count = 0;
  for (let i = 0; i < num; i++){
    for(let j = 0; j < num; j++){
      if(grid_v[i][j]){
        count += 1;
      }
    }
  }
  if(count >= num*num){
    noLoop();
    console.log("Bingo!!!");
  }
}

function draw_grid(){
  noStroke();
  for (let i = 0; i < num; i++){
    for(let j = 0; j < num; j++){
      let current_grid_v = grid_v[i][j];
      if (current_grid_v){
        fill('#71B8A9');
      } else if (!current_grid_v){
        fill('#efefef');
      }
      rect(w_grid*i, h_grid*j, w_grid, h_grid);
    }
  }
}
