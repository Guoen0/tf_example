let num = 5;
let w_grid,h_grid;
let RL = new DQN();
let env = new Environment(num);
let step = 0;
function setup() {
  createCanvas(500,500);
  frameRate(60);
  w_grid = width/num;
  h_grid = height/num;
  env.init_grid();

}

function draw() {
  background(51);

  if (step == 0){
    env.get_current_state();
    env.draw_grid();
  }
  //env.action_random(); // random paly
  //env.draw_grid(); // if human play, enable this line

  // run
  env.step(RL.choose_action(env.state));
  RL.store_transition(env.state, env.next_state ,env.action ,env.reward);
  env.state = env.next_state;
  // train
  if (step > 200 && step % 5 == 0){
    RL.learn();
  }


  step += 1;
  if(env.is_finish){
    env.init_grid();
  }


  //console.log(env.reward);
}


/*
// Human play
function mousePressed(){
  let index_x = floor(mouseX/w_grid);
  let index_y = floor(mouseY/h_grid) + 1;
  env.step(index_x + index_y*num);
}
*/
