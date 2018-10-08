class DQN{
  constructor(){
    this.lr = 0.01;
    this.gamma = 0.9;
    this.epsilon = 0.9;
    this.replace_target_iter = 300;

    this.features_num = num*num;
    this.action_num = num*num;
    this.memory_size = 500;
    this.batch_size = 32;
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
    this.q_targets = new Array();
    this.q_eval_wrt_a_s = new Array();
    this.learn_step_counter = 0;
    this.memory_counter = 0;


    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[this.features_num]});
    let e1 = tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);
    this.q_eval = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(e1);
    this.eModel = tf.model({inputs:this.xs, outputs:this.q_eval});


    let t1 = tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);
    this.q_next = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(t1);
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

  replace_target_params(){
    // Cannot find get_collection method
  }

  learn(){
    if(this.learn_step_counter & this.replace_target_iter == 0){
      this.replace_target_params();
    }

    let sample_indexs = new Array();
    let batch_memory = new Array();
    let ss = new Array();
    let s_s = new Array();
    let as = new Array();
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
      //as.push(this.memory[sample_indexs[i]].slice(this.features_num*2, this.features_num*2+1));
      this.a.push(this.memory[sample_indexs[i]].slice(this.features_num*2, this.features_num*2+1))
      rs.push(this.memory[sample_indexs[i]].slice(this.features_num*2+1, this.features_num*2+2));
    }
    this.s = tf.tensor(ss,[this.batch_size,this.features_num]);
    this.s_ = tf.tensor(s_s,[this.batch_size,this.features_num]);
    //this.a = tf.tensor(as,[this.batch_size,1]);
    this.r = tf.tensor(rs,[this.batch_size,1]);
    this.train();
  }

  async train(){ //No translation yet
    for(let i = 0; i < this.batch_size; i++){
      const q_target = this.tModel.predict( this.s_.gather([i]) ).max(1).mul(this.gamma).add(this.r);
      this.q_targets.push(q_target);

      //const a_indices = tf.stack([tf.range(0,this.a.shape[0]), this.a], 1);
      this.q_eval_wrt_a_s.push( this.eModel.predict( this.s.gather([i]) ).gather(this.a[i]) );
    }
    //console.log(this.q_targets);
    this.q_target = tf.stack(this.q_targets);
    this.q_eval_wrt_a = tf.stack(this.q_eval_wrt_a_s);
    this.loss = tf.mean( tf.squaredDifference(this.q_target, this.q_eval_wrt_a));
    tf.train.rmsprop(this.lr).minimize(this.loss);
    this.q_targets = new Array();
    this.q_eval_wrt_a_s = new Array();
  }


  //this.epsilon = this.epsilon; // unfinish

  //this.learn_step_counter += 1;
}
