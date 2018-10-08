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
    this.a;
    this.r;
    this.xs;
    this.ys;
    this.action;

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


    /*  //Do not understand
    const q_target = this.r + this.gamma * this.q_next.max(1);
    this.q_target = q_target;
    const a_indices = tf.stack([tf.range(0,this.a.shape[0]), this.a], 1);
    this.q_eval_wrt_a = q_eval.gather(a_indices);
    this.loss = tf.mean( tf.squaredDifference(this.q_target, this.q_eval_wrt_a));
    this._train_op = tf.train.rmsprop(this.lr).minimize(this.loss);
    */
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
        let actions_value = this.eModel.predict(observation);
        this.action = tf.argMax(actions_value).dataSync();// Unsolved
      } else {
        this.action = floor(radom(this.action_num - 0.001));
      }


    });
    this.action;
    console.log(this.action);
  }

  replace_target_params(){
    // Cannot find get_collection method
  }

  learn(){
    if(this.learn_step_counter & ths.replace_target_iter == 0){
      this.replace_target_params();
    }

    let sample_indexs = new Array();
    let batch_memory = new Array();
    let ss,s_s,as,rs = new Array();
    if(this.memory_counter > this.memory_size){
      for (let i = 0; i < this.batch_size; i++){
        let sample_index = random(this.memory_size);
        sample_indexs.push(sample_index);
      }
    } else {
      for (let i = 0; i < this.batch_size; i++){
        let sample_index = random(this.memory_size);
        sample_indexs.push(sample_index);
      }
    }
    for (let i = 0; i < this.batch_size; i++){
       ss.push(this.memory.slice(0,this.features_num));
       s_s.push(this.memory.slice(this.features_num, this.features_num*2));
       as.push(this.memory.slice(this.features_num*2));
       rs.push(this.memory.slice(this.features_num*2+1));
    }
    this.s = tf.tensor(ss,[this.batch_size,this.features_num]);
    this.s_ = tf.tensor(s_s,[this.batch_size,this.features_num]);
    this.a = tf.tensor(as,[this.batch_size,1]);
    this.r = tf.tensor(rs,[this.batch_size,1]);
    train();
  }

  async train(){ //No translation yet

  }


  //this.epsilon = this.epsilon; // unfinish

  //this.learn_step_counter += 1;
}
