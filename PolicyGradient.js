class PolicyGradient{
  constructor(){
    this.memory = [];
    this.action = [0,0];
    this.xs;
    this.ys;
    this.model;
    //this.build_net();
  }



  build_net(){

  }


  choose_aciton(){
    this.memory = [];
    let index_x = floor(random(num-0.0001));
    let index_y = floor(random(num-0.0001));
    this.action = [index_x, index_y];
  }




  learn(s, r){
    this.memory.push(s, r);
    //console.log(this.memory);
  }

}
