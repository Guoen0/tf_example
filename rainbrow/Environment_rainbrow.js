class Environment{
  constructor(){
    this.enemy_num = num;
    this.enemy_params = 4;
    this.grid = this.enemy_num*2-1;
    this.grid_height = height/this.grid;
    this.enemies = new Array(this.enemy_num);
    for (let i = 0; i < this.enemy_num; i++){
      this.enemies[i] = new Array(this.enemy_params);
    }
    this.enemy_height = this.grid_height*0.82;
    this.enemy_width = this.enemy_height;

    this.food = new Array(3);
    this.food_width = this.grid_height*0.2;
    this.agent = new Array(2);
    this.agent_width = this.grid_height*0.5;

    this.reward = 0;
    this.state = new Array();
    this.state_next = new Array();
    this.satiety = 0;
    this.goal = 0;
    this.maxGoal = 0;
    this.action = 0;
    this.action_count = 0;
    this.is_dead = false;
    this.death = 0;
  }
  init(){
    for (let i = 0; i < this.enemy_num; i++){
      this.enemies[i][0] = random(this.enemy_width/2, width-this.enemy_width/2);
      //this.enemies[i][0] = this.enemy_width/2;
      this.enemies[i][1] = this.grid_height/2 + i*this.grid_height*2;

      let dir;
      if( random(1) < 0.5 ){ dir = 1 } else { dir = -1 };
      this.enemies[i][2] =  this.get_random_speed() * dir;

    }
    this.agent[0] = width/2;
    this.agent[1] = floor(random(this.enemy_num-1.001))*this.grid_height*2 + this.grid_height*1.5;
    //console.log(this.agent[1]);
    this.init_food();
    this.is_dead = false;
    this.action_count = 0;
    this.reward = 0;
    this.satiety = 0;
    this.goal = 0;
    this.judgment();
    this.get_state_next();
    this.state = this.state_next;
  }
  get_random_speed(){
    return random(width/60/4, width/60/6);
  }
  init_food(){
    this.food[0] = width/2;
    this.food[1] = floor(random(this.enemy_num-0.001))*this.grid_height*2 + this.grid_height/2;
  }

  update(){
    this.move_enemies();
    this.move_agent();
    this.edge();
    this.judgment();
    this.get_state_next();
    this.draw();

    if(this.maxGoal <= this.goal){this.maxGoal = this.goal};
  }
  move_enemies() {
    for (let i = 0; i < this.enemy_num; i++){
      this.enemies[i][0] += this.enemies[i][2];
    }
  }
  move_agent(){
    this.agent[1] += (this.action-0.5)*2 * width/60/1;
  }

  edge(){
    for (let i = 0; i < this.enemy_num; i++){
      if (this.enemies[i][0] <= this.enemy_width/2){
        this.enemies[i][0] = this.enemy_width/2;
        this.enemies[i][2] = this.get_random_speed();
      }
      if (this.enemies[i][0] >= width-this.enemy_width/2){
        this.enemies[i][0] = width-this.enemy_width/2;
        this.enemies[i][2] = - this.get_random_speed();
      }
    }
    if(this.agent[1] <= this.agent_width/2){
      this.agent[1] = this.agent_width/2;
    }
    if(this.agent[1] >= height-this.agent_width/2){
      this.agent[1] = height-this.agent_width/2;
    }
  }


  step(a){
    this.action = a;
    this.action_count += 1;
  }

  get_state_next(){
    this.state_next = new Array();
    for (let i = 0; i < this.enemy_num; i++){

        this.state_next.push(this.enemies[i][3]);

    }
    this.state_next.push(this.food[2]);
    this.state_next.push(this.agent[1]);
  }

  ///////////////////////////////////////////////////////////////////////
  judgment(){
    let dist_f_abs = abs(this.food[1] - this.agent[1]);
    let dist_f = this.food[1] - this.agent[1];
    this.food[2] = dist_f;
    this.reward = (0 - dist_f_abs/height)*0.1;
    if(dist_f_abs <= this.food_width/2+this.agent_width/2){
      this.reward = 0.5;
      this.satiety += 10;
      this.goal += 1;
      this.init_food();
    }

    for(let i = 0; i < this.enemy_num; i++){
      let dist_e = pow((this.enemies[i][0] - this.agent[0]), 2) + pow((this.enemies[i][1] - this.agent[1]), 2);
      this.enemies[i][3] = dist_e;
      if(dist_e <= pow(this.enemy_width/2+this.agent_width/2, 2)){
        this.reward = -0.2;
        this.go_dead();
      }
    }

    if(this.satiety <= -10){
      this.go_dead();
    } else if(this.satiety > -10){
      this.satiety -= 0.02;
    }
    if(this.satiety > 10){
      this.satiety = 10;
    }

  }
/////////////////////////////////////////////////////////////////////////
  go_dead(){
    this.is_dead = true;
    this.death += 1;
    background('red');
    //console.log('YOU DEAD!!!');
  }

  draw(){
    noStroke();
    //background(51);
    for (let i = 0; i < this.grid; i++){
      fill(map(i,0,this.grid,195,275), map(i,0,this.grid,230,100),map(i,0,this.grid,340,80));
      rect(0, i*this.grid_height, width, (i+1)*this.grid_height);
    }

    fill('#9DF792');
    ellipse(this.food[0],this.food[1],this.food_width,this.food_width);
    for (let i = 0; i < this.enemy_num; i++){
      let x = this.enemies[i][0];
      let y = this.enemies[i][1];

      fill('#F02C28');
      ellipse(x,y,this.enemy_width,this.enemy_height);
    }
    fill('#FfeA65');
    ellipse(this.agent[0],this.agent[1],this.agent_width,this.agent_width);

  }
}
