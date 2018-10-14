let num = 3;

let env;
let step = 0;
let game_step = 2;
let start_l_step = 10;

let lr = 0.1;
let GAMMA = 0.9;
let epsilon = 0.9;
let features_num = num*4 + 2 + 1;
let action_num = 3;
let units_num = 4;

RL_A = new Actor();
RL_C = new Critic();
let g_td_error;

let rewardP;
let lifeP;
let maxLifeP;
let deathP;
let td_error_P;
let button_MP;
let button_RP;
let button_dontL;
let button_S;
let is_human = false;
let is_stop = false;
let is_justRun = false;
let is_train = true;

let fr = 30;
function setup() {
  createCanvas(500,500);
  colorMode(HSB,360);
  frameRate(fr);
  env = new Environment();
  env.init();
  env.update();

  button_MP = createButton("Human try");
  button_MP.mousePressed(human);
  button_RP = createButton("Train robot");
  button_RP.mousePressed(train_robot);
  button_dontL = createButton("just RUN");
  button_dontL.mousePressed(just_run);
  button_S = createButton("STOP");
  button_S.mousePressed(stop_);
  rewardP = createP("Reward: 0");
  lifeP = createP("Life: 0");
  maxLifeP = createP("MaxLife: 0");
  deathP = createP("Death: 0");
  td_error_P = createP("TD_error: 0");
}

function draw() {
  if (!is_stop){


    if(!is_human){
      if(step % game_step == 0){
        env.step(RL_A.choose_action(env.state));
      }
    }


    env.update();


    if(is_train){
      if( step > start_l_step && step % game_step == 0){
        g_td_error =  RL_C.learn(env.state, env.state_next, env.reward);
        RL_A.learn(env.state, env.action, g_td_error);
      }
    }

    env.state = env.state_next;

    if(env.is_dead){
      env.init();
    }


    // print
    rewardP.html("Reward: " + env.reward.toFixed(1));
    lifeP.html("Life: " + env.life.toFixed(1));
    maxLifeP.html("MaxLife: " + env.maxLife.toFixed(1));
    deathP.html("Death :" + env.death);
    td_error_P.html("TD_error :" + g_td_error);
    step += 1;
  }

}



function human(){
  is_human = true;
  is_stop = false;
  is_justRun = false;
  is_train = false;
}
function train_robot(){
  is_human = false;
  is_stop = false;
  just_run = false;
  is_train = true;
}
function stop_(){
  is_stop = true;
  is_justRun = false;
  is_train = false;
}
function just_run(){
  is_human = false;
  is_justRun = true;
  is_train = false;
}

// Human play
function keyPressed(){
  if(is_human){
    if(keyCode === UP_ARROW){
      env.step(0);
    } else if (keyCode === DOWN_ARROW){
      env.step(2);
    }
  }
}
function keyReleased(){
  if(is_human){
    env.step(1);
  }
}
