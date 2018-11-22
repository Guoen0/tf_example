class RL_{
  constructor(){
    this.features_num = 5;
    this.outputs_num = 9;
    this.units_num = 32;
    this.activation_a = 'elu';
    this.activation_a_last = '';
    this.lr = 0.01;
    this.optimaizer = tf.train.adam(this.lr);
    this.xs;
    this.outputs;
    this.inputs;
    this.labels;
    this.loss;
    this.build_network();
  }

  build_network(){
    this.xs = tf.input({shape:[this.features_num]});
    let h1 = tf.layers.dense({
      units: this.units_num,
      activation: this.activation_a,
      inputShape: [this.features_num],
    }).apply(this.xs);

    let h2 = tf.layers.dense({ units: this.units_num, activation: this.activation_a, }).apply(h1);
    let h3 = tf.layers.dense({ units: this.units_num, activation: this.activation_a, }).apply(h2);
    let h4 = tf.layers.dense({ units: this.units_num, activation: this.activation_a, }).apply(h3);

    this.output_layer = tf.layers.dense({
      units: this.outputs_num,
      //actication: this.activation_a_last,
    }).apply(h4);

    this.pModel = tf.model({inputs:this.xs, outputs:this.output_layer});
  }

  learn(_input,_label){
    tf.tidy(() => {
      this.inputs = tf.tensor(_input, [1, this.features_num]);
      this.labels = tf.tensor(_label, [1, this.outputs_num]);
      this.train();
    });
  }

  get_loss(_output){
    const loss = tf.sub(_output,this.labels).square().mean();
    this.loss = loss.dataSync()[0];
    loss_gui =  this.loss;
    //loss.print();
    return loss;
  }

  async train(){
    this.optimaizer.minimize(() => this.get_loss(this.predict_v(this.inputs)));
  }

  predict_v(inp){
    const v = this.pModel.predict(inp);
    return v;
  }
}
