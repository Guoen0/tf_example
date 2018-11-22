let data_input = new Array();
let data_label = new Array();
RL = new RL_();

let predict_v_gui;
let loss_gui;
function setup() {
  createCanvas(500,500);
  frameRate(30);

  for (let i = 0; i < 100; i ++){
    let inp = new Array();
    let lab = new Array();
    for (let j = 0; j < 5; j++){
      inp.push(random(0,10));
    }
    for (let j = 0; j < 9; j++){

      if(j>=5){
        lab.push(random(100,200));
      } else {
        lab.push(random(300,400));
      }
    }
    data_input.push(inp);
    data_label.push(lab);
  }
}

function draw() {
  background(51);


  let index = floor(random(100));
  RL.learn(data_input[index], data_label[index]);

  tf.tidy(() => {
    let xx = new Array();
    for (let i = 0; i < 5; i++){
      xx.push(random(0,10));
    }
    let xxx = tf.tensor(xx,[1,5]);
    let yyy = RL.predict_v(xxx);
    predict_v_gui = yyy.dataSync();

  });



  fill(255);
  for(let i = 0; i < 9; i++){
    rect(i*50,0, 50, 50+predict_v_gui[i]);
  }
  fill('red');
  rect(0,0,loss_gui/10,50);
  //console.log("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
}
