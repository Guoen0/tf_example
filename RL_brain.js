class DQN{
  constructor(){
    this.lr = 0.01;
    this.gamma = 0.9;
    this.epsilon = 0.85;
    this.replace_target_iter = 64;

    this.features_num = num*num;
    this.action_num = num*num;
    this.memory_size = 1000;
    this.batch_size = 64;
    this.memory = new Array(this.memory_size);
    for (let i = 0; i < this.memory_size; i++){
      this.memory[i] = new Array(this.features_num*2+2);
    }

    this.s;
    this.s_;
    this.a = new Array();
    this.r;
    this.xs;
    this.ys;
    this.action;
    this.learn_step_counter = 0;
    this.memory_counter = 0;
    this.saveModel;
    this.optimizer = tf.train.rmsprop(this.lr);
    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[this.features_num]});
    let e1 = tf.layers.dense({
      units: 128,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);

    let e2 = tf.layers.dense({
      units: 32,
      activation: 'relu',
    }).apply(e1);

    this.q_eval = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(e2);

    this.eModel = tf.model({inputs:this.xs, outputs:this.q_eval});


    let t1 = tf.layers.dense({
      units: 128,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);
    let t2 = tf.layers.dense({
      units: 32,
      activation: 'relu',
    }).apply(t1);
    this.q_next = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(t2);
    this.tModel = tf.model({inputs:this.xs, outputs:this.q_next});

  }

  store_transition(s, s_, a, r){
    let transition = s.concat(s_).concat(a).concat(r);
    let index = this.memory_counter % this.memory_size
    this.memory[index] = transition;
    this.memory_counter += 1;
  }

  choose_action(observation_){
    tf.tidy(() => {
      let observation = tf.tensor([observation_]);
      if (random(1) < this.epsilon){
        let actions_value = this.eModel.predict(observation,{batchSize: 1});
        this.action = tf.argMax(actions_value, 1).dataSync()[0];
      } else {
        this.action = floor(random(this.action_num - 0.001));
      }
    });
    return this.action;
  }

  async replace_target_params(){
    this.saveModel = await this.eModel.save('localstorage://my-model-1');
    this.tModel = await tf.loadModel('localstorage://my-model-1');
  }

  learn(){
    if(this.learn_step_counter % this.replace_target_iter == 0){
      this.replace_target_params();
    }

    let sample_indexs = new Array();
    let batch_memory = new Array();
    let ss = new Array();
    let s_s = new Array();
    let as = new Array();
    this.a = new Array();
    let rs = new Array();

    if(this.memory_counter > this.memory_size){
      for (let i = 0; i < this.batch_size; i++){
        let sample_index = floor(random(this.memory_size-0.001));
        sample_indexs.push(sample_index);
      }
    } else {
      for (let i = 0; i < this.batch_size; i++){
        let sample_index = floor(random(this.memory_counter-0.001));
        sample_indexs.push(sample_index);
      }
    }
    for (let i = 0; i < this.batch_size; i++){
      ss.push(this.memory[sample_indexs[i]].slice(0,this.features_num));
      s_s.push(this.memory[sample_indexs[i]].slice(this.features_num, this.features_num*2));
      this.a.push(this.memory[sample_indexs[i]].slice(this.features_num*2, this.features_num*2+1))
      rs.push(this.memory[sample_indexs[i]].slice(this.features_num*2+1, this.features_num*2+2));
    }
    tf.tidy(() => {
      this.s = tf.tensor(ss,[this.batch_size,this.features_num]);
      this.s_ = tf.tensor(s_s,[this.batch_size,this.features_num]);
      this.r = tf.tensor(rs,[this.batch_size,1]);
      this.train();
    });

    //this.epsilon = this.epsilon; //
    this.learn_step_counter += 1;
  }


  loss(q_target, q_eval_wrt_a){
      const TD_error = tf.mean( tf.squaredDifference(q_target, q_eval_wrt_a) );
      //TD_error.print();
      return TD_error;
  }

  predict_q_target(s_, i){
    const q_target = this.tModel.predict( s_.gather([i]) ).max(1).mul(this.gamma).add(this.r.gather([i]));
    return q_target;
  }

  async train(){
    for(let i = 0; i < this.batch_size; i++){
      const q_eval_ = this.eModel.predict( this.s.gather([i]) );
      this.q_eval_wrt_a = q_eval_.gather(this.a[i],1).reshape([1,1]);
      this.optimizer.minimize(() => this.loss(this.predict_q_target(this.s_, i), this.q_eval_wrt_a));
    }
  }

}
