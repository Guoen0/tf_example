

class Actor{
  constructor(){
    this.action;
    this.action_prb = new Array(action_num);
    this.s;
    this.a;
    this.TD_error;
    this.xs;
    this.exp_v;
    this.optimizer = tf.train.adam(lr);
    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[features_num]});

    let h1 = tf.layers.dense({
      units: units_num,
      activation: activation,
      inputShape: [features_num],
    }).apply(this.xs);

    let h2 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h1);
    let h3 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h2);
    let h4 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h3);

    this.acts_prob = tf.layers.dense({
      units: action_num,
      activation: 'softmax',
    }).apply(h4);
    this.aModel = tf.model({inputs:this.xs, outputs:this.acts_prob});

  }

  choose_action(observation_){
    tf.tidy(() =>{
      let observation = tf.tensor([observation_]);
      let probs = this.predict_acts_prob(observation);
      //console.log("probs:")
      this.action_prb[0] = probs.dataSync()[0];
      this.action_prb[1] = probs.dataSync()[1];
      if(goal<50){
        this.action = tf.multinomial(probs,1).dataSync()[0];
      }else{
        this.action = tf.multinomial(tf.mul(probs,10),1).dataSync()[0];
      }
    });
    return this.action;
  }

  learn(s, a, td){
    tf.tidy(() =>{
      this.s = tf.tensor(s,[1,features_num]);
      this.a = a;
      this.TD_error = tf.tensor([td]);
      //this.s.print();
      //console.log("action: "+this.a);
      this.train();
    });
  }

  predict_acts_prob(s){
    return  this.aModel.predict( s );
  }
  get_exp_v(s){
    let indices = tf.tensor1d([this.a], 'int32');
    //console.log("indices:");
    //indices.print();
    const log_prob = tf.log( this.predict_acts_prob( s ).gather(indices,1).add(0.00000000001) );
    const exp_v = log_prob.mul(this.TD_error).mean().mul(-1);
    this.exp_v = exp_v.dataSync()[0];
    return exp_v;
  }

  async train(){
    this.optimizer.minimize(() => this.get_exp_v( this.s ));
  }
}



class Critic{
  constructor(){
    this.TD_error = tf.scalar(0);
    this.loss_value;
    this.s;
    this.s_;
    this.v;
    this.v_;
    this.r;
    this.xs;
    this.optimizer = tf.train.adam(lr);
    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[features_num]});
    let h1 = tf.layers.dense({
      units: units_num,
      activation: activation,
      inputShape: [features_num],
    }).apply(this.xs);

    let h2 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h1);
    let h3 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h2);
    let h4 = tf.layers.dense({ units: units_num, activation: activation, }).apply(h3);

    this.v_output_layer = tf.layers.dense({
      units: 1,
      //activation: 'softmax',
    }).apply(h4);
    this.cModel = tf.model({inputs:this.xs, outputs:this.v_output_layer});

  }

  learn(s, s_, r){
    tf.tidy(() => {
      this.s = tf.tensor(s,[1,features_num]);
      this.s_ = tf.tensor(s,[1,features_num]);
      this.r = tf.tensor([r], [1,1]);
      this.train();
    });
    return this.TD_error;
  }

  async train(){
    //console.log("s_:");
    //console.log(this.predict_v(this.s_));
    const v_ = this.predict_v(this.s_);
    this.v_v = v_.dataSync()[0];
    this.optimizer.minimize(() => this.loss(this.predict_v( this.s ), v_));
  }
  loss(v, v_){
    //console.log("v_:");
    //v_.print();
    const loss = this.get_td_error(v, v_).square().mean();
    return loss;
  }
  get_td_error(v, v_){
    const td_error = tf.mul(v_, GAMMA).sub(v).add(this.r);
    this.TD_error = td_error.dataSync()[0];
    return td_error;
  }
  predict_v(s){
    const v = this.cModel.predict( s );
    return v;
  }


}
