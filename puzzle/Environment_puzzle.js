class Environment{
  constructor(){
    this.grid_v = new Array(num);
    for (let i = 0; i < num; i++){
      this.grid_v[i] = new Array(num);
    }
    this.state = new Array(num*num);
    this.next_state = new Array(num*num);
    this.reward;
    this.action;
    this.action_count = 0;
    this.is_finish = false;
  }

  step(a){
    this.action = a;
    let index_x = int(a % num);
    let index_y = floor(a/num);
    this.turn(index_x, index_y);
    this.judgment();
    this.draw_grid();
    this.action_count += 1;
    //console.log(this.action_count);
  }

  action_random(){
    let index_x = floor(random(num-0.0001));
    let index_y = floor(random(num-0.0001));
    this.turn(index_x, index_y);
    this.judgment();
    this.draw_grid();
    this.action_count += 1;
    console.log(this.action_count);
  }

  turn(x, y){
    this.grid_v[x][y] = !this.grid_v[x][y];
    if(x+1 < num){ this.grid_v[x+1][y] = !this.grid_v[x+1][y] };
    if(x-1 >= 0){ this.grid_v[x-1][y] = !this.grid_v[x-1][y] };
    if(y+1 < num){ this.grid_v[x][y+1] = !this.grid_v[x][y+1] };
    if(y-1 >= 0){ this.grid_v[x][y-1] = !this.grid_v[x][y-1] };
  }
  init_grid(){
    for (let n = 0; n < 1; n++){
      for (let i = 0; i < num; i++){
        for(let j = 0; j < num; j++){
          if(random(1)<0.5){
            this.turn(i, j);
          }
        }
      }
    }
    this.is_finish = false;
    this.action_count = 0;
  }
  judgment(){
    let count = 0;
    for (let i = 0; i < num; i++){
      for(let j = 0; j < num; j++){
        if(this.grid_v[i][j]){
          count += 1;
          this.next_state[i + j*num] = 1;
        } else {
          this.next_state[i + j*num] = 0;
        }
      }
    }
    this.reward = pow(count,2)/(pow(num,3));
    // Intervention
    if (this.action_pre == this.action){
      this.reward = - 1;
    }
    if(count >= num*num){
      this.victory();
      this.reward = 20;
    }
  }
  get_current_state(){
    for (let i = 0; i < num; i++){
      for(let j = 0; j < num; j++){
        if(this.grid_v[i][j]){
          this.state[i + j*num] = 1;
        } else {
          this.state[i + j*num] = 0;
        }
      }
    }
  }
  draw_grid(){
    noStroke();
    for (let i = 0; i < num; i++){
      for(let j = 0; j < num; j++){
        let current_grid_v = this.grid_v[i][j];
        if (current_grid_v){
          fill('#71B8A9');
        } else if (!current_grid_v){
          fill('#efefef');
        }
        rect(w_grid*i, h_grid*j, w_grid, h_grid);
      }
    }
  }
  victory(){
    //noLoop();
    console.log("YOU WIN!!!");
    this.is_finish = true;
  }
}
