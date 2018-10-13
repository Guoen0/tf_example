class DQN{
  constructor(){
    this.lr = 0.005;
    this.gamma = 0.9;
    this.epsilon = 0.9;
    this.features_num = num*3 + 2;
    this.action_num = 3;
    this.batch_size = 2;
    this.iteration = 8;
    this.replace_target_iter = 16;
    this.memory_size = 10000;
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
    this.optimizer = tf.train.adam(this.lr);
    this.TD_error = 0;
    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[this.features_num]});
    let e1 = tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);
    let e2 = tf.layers.dense({
      units: 16,
      activation: 'relu',
    }).apply(e1);
    let e3 = tf.layers.dense({
      units: 16,
      activation: 'relu',
    }).apply(e2);
    let e4 = tf.layers.dense({
      units: 16,
      activation: 'relu',
    }).apply(e3);
    let e5 = tf.layers.dense({
      units: 8,
      activation: 'relu',
    }).apply(e4);

    this.q_eval = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(e5);
    this.eModel = tf.model({inputs:this.xs, outputs:this.q_eval});


    let t1 = tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [this.features_num],
    }).apply(this.xs);
    let t2 = tf.layers.dense({
      units: 16,
      activation: 'relu',
    }).apply(t1);
    this.q_next = tf.layers.dense({
      units: this.action_num,
      activation: 'sigmoid',
    }).apply(t2);
    this.tModel = tf.model({inputs:this.xs, outputs:this.q_next});

  }

  store_transition(s, s_, a, r){
    let tf_s = tf.tensor1d(s);
    let transition = tf_s.concat(s_).concat([a]).concat([r]);
    transition.reshape([this.features_num*2+2,1]);
    let index = this.memory_counter % this.memory_size
    this.memory[index] = transition;
    this.memory_counter += 1;
  }

  choose_action(observation_){
    tf.tidy(() => {
      let observation = tf.tensor([observation_]);

      if (step > start_l_step && random(1) < this.epsilon){
        let actions_value = this.eModel.predict(observation,{batchSize: 1});
        this.action = tf.argMax(actions_value, 1).dataSync()[0];
      } else {
        this.action = floor(random(this.action_num - 0.001));
      }
    });
    return this.action;
  }

  async replace_target_params(){
    this.learn_step_counter += 1;
    if(this.learn_step_counter % this.replace_target_iter == 0){
      this.saveModel = await this.eModel.save('localstorage://my-model-1');
      this.tModel = await tf.loadModel('localstorage://my-model-1');
      //console.log('replace');
    }
  }

  learn(){
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
      s_s.push(this.memory[sample_indexs[i]].slice(this.features_num,this.features_num));
      this.a.push(this.memory[sample_indexs[i]].slice(this.features_num*2, 1))
      rs.push(this.memory[sample_indexs[i]].slice(this.features_num*2+1, 1));
    }

    tf.tidy(() => {
      let sst = tf.concat(ss);
      let s_st = tf.concat(s_s);
      let rst = tf.concat(rs);
      this.s = sst.reshape([this.batch_size,this.features_num]);
      this.s_ = s_st.reshape([this.batch_size,this.features_num]);
      this.r = rst.reshape([this.batch_size,1]);
      this.train();
    });

    //this.epsilon = this.epsilon; //
  }


  loss(q_target, q_eval_wrt_a){
      const TD_error = tf.mean( tf.squaredDifference(q_target, q_eval_wrt_a) );
      this.TD_error = TD_error.dataSync()[0];
      //console.log(TD_error);
      return TD_error;
  }

  predict_q_eval_wrt_a(s){
    let q_eval_wrt_as = [];
    for(let i = 0; i < this.batch_size; i++){
      const q_eval = this.eModel.predict( s ).gather([i]);
      const indeices = tf.tensor1d([this.a[i]], 'int32');
      const q_eval_wrt_a = q_eval.gather(indeices);
      q_eval_wrt_as.push(q_eval_wrt_a);
    }
    const q_eval_wrt_a_tensor = tf.concat(q_eval_wrt_as);
    return q_eval_wrt_a_tensor;
  }

  async train(){
    let q_targets = [];
    const q_target = this.tModel.predict( this.s_ ).max(1).mul(this.gamma).reshape([this.batch_size,1]).add(this.r);
    this.optimizer.minimize(() => this.loss(q_target, this.predict_q_eval_wrt_a(this.s)));
  }

}
