class Actor{
  constructor(){
    this.action;

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
      activation: 'relu',
      inputShape: [features_num],
    }).apply(this.xs);
    let h2 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h1);
    let h3 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h2);
    let h4 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h3);
    let h5 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h4);

    this.acts_prob = tf.layers.dense({
      units: action_num,
      activation: 'softmax',
    }).apply(h5);
    this.aModel = tf.model({inputs:this.xs, outputs:this.acts_prob});

  }

  choose_action(observation_){
    tf.tidy(() =>{
      let observation = tf.tensor([observation_]);
      let probs = this.predict_acts_prob(observation);
      probs.print();
      this.action = tf.multinomial(probs,1).dataSync()[0];
    });
    return this.action;
  }

  learn(s, a, td){
    tf.tidy(() =>{
      this.s = tf.tensor(s,[1,features_num]);
      this.a = a;
      this.TD_error = tf.tensor([td]);
      this.train();
    });
  }

  predict_acts_prob(s){
    return  this.aModel.predict( s );
  }
  get_exp_v(s){
    let indices = tf.tensor1d([this.a], 'int32');
    const log_prob = tf.log( this.predict_acts_prob( s ).gather(indices,1) );
    this.exp_v = log_prob.mul(this.TD_error).mean().mul(-1);
    return this.exp_v;
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
      activation: 'relu',
      inputShape: [features_num],
    }).apply(this.xs);
    let h2 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h1);
    let h3 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h2);
    let h4 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h3);
    let h5 = tf.layers.dense({
      units: units_num,
      activation: 'relu',
    }).apply(h4);

    this.v_output_layer = tf.layers.dense({
      units: 1,
      //activation: 'softmax',
    }).apply(h5);
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
    const v_ = this.predict_v(this.s_);
    this.optimizer.minimize(() => this.loss(this.predict_v( this.s ), v_));
  }
  loss(v, v_){
    const loss = this.get_td_error(v, v_).square().mean();
    return loss;
  }
  get_td_error(v, v_){
    const td_error = v_.mul(GAMMA).sub(v).add(this.r);
    this.TD_error = td_error.dataSync()[0];
    return td_error;
  }
  predict_v(s){
    return this.cModel.predict( s );
  }


}
