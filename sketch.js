let num = 4;
let w_grid,h_grid;
let RL = new DQN();
let env = new Environment(num);
let step = 0;
let start_l_step = 500;


let lossP;
let rewardP;
let epsilonSlider;
function setup() {
  createCanvas(500,500);
  frameRate(60);
  w_grid = width/num;
  h_grid = height/num;
  env.init_grid();

  lossP = createP('loss');
  rewardP = createP('Reward');
  epsilonSlider = createSlider(0, 100);
}

function draw() {
  interaction();
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
  RL.replace_target_params();
  env.state = env.next_state;
  rewardP.html('Reward: ' + env.reward);
  // train
  if (step > start_l_step && step % 1 == 0){
    for(i = 0; i < RL.iteration; i++){
      RL.learn();
    }
  }

  console.log('step');
  step += 1;
  if(env.is_finish){
    env.init_grid();
  }


  //console.log(env.reward);
}

function interaction(){
  RL.epsilon = epsilonSlider.value()/100;
}

/*
// Human play
function mousePressed(){
  let index_x = floor(mouseX/w_grid);
  let index_y = floor(mouseY/h_grid);
  env.step(index_x + index_y*num);
}
*/
