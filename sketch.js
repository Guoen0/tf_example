let num = 8;
let w_grid,h_grid;
let RL = new PolicyGradient();
let env = new Environment(num);
function setup() {
  createCanvas(500,500);
  frameRate(5);
  w_grid = width/num;
  h_grid = height/num;
  env.init_grid();
}

function draw() {
  background(51);
  //env.action_random();
  RL.choose_aciton();
  env.action(RL.action);
  env.judgment();
  env.draw_grid();
  RL.learn(env.state, env.reward);
}



function mousePressed(){
  let index_x = floor(mouseX/w_grid);
  let index_y = floor(mouseY/w_grid);
  env.turn(index_x, index_y);
}
