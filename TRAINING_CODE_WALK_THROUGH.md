# Code Walkthroguh
트레이닝 시 코드의 각 부분에서 어떤 기능들을 수행하는지 가이드 하여 코드의 흐름 파악을 용이하게 하고자 함.

### train_local.py 흐름
트레이닝은 [train_local.py](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py) 스크립트를 사용하여 실행.

train_local.py 는 다음 기능들을 수햏한다.
1. [L8](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L8): [flags.py]() 에서 옵션들을 읽어온다.
1. [L9](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L9): 데이터 파일이 필요한 경우 다운로드 한다.
1. [L10](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train_local.py#L10): [Trainer.train()](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L36) 함수를 실행한다.

```python
    # option들 읽어오기
    options = get_options_from_flags()
    
    # 데이터 파일 다운로드 하기
    maybe_download_data_files_from_s3(options)
    
    # `Trainer`의 `train()` 함수를 실행
    Trainer(options).train()
```

### Trainer.train() 흐름
1. [L41](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L41):
cpu를 사용하여 아래의 작업들을 실행함

    ```python
    with tf.Graph().as_default(), tf.device('/cpu:0'):
    ```

1. [L45-L61](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L45-L61):
word_embedding, char_embedding, learning_rate, optimizer 에 대한 Tensorflow Varaible 설정

    ```python
    embedding_placeholder = tf.placeholder(tf.float32,
        shape=self.sq_dataset.embeddings.shape)
    embedding_var = \
        tf.Variable(embedding_placeholder, trainable=False)
    word_chars_placeholder = tf.placeholder(tf.float32,
        shape=self.sq_dataset.word_chars.shape)
    word_chars_var = \
        tf.Variable(word_chars_placeholder, trainable=False)
    
    learning_rate = tf.Variable(name="learning_rate", initial_value=
        self.options.learning_rate, trainable=False, dtype=tf.float32)
    learning_rate_placeholder = tf.placeholder(tf.float32)
    assign_learning_rate = tf.assign(learning_rate,
            tf.maximum(self.options.min_learning_rate,
                learning_rate_placeholder))
    self.optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    ```

1. [L62-L64](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L62-L64):
모델 초기화

    ```python
    self.model_builder = ModelBuilder(self.optimizer, self.options,
        self.sq_dataset, embedding_var, word_chars_var,
        compute_gradients=True, sess=self.session)
    ```

1. [L73-L75](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L73-L75):
그레디언트를 초깃값들을 가져와서, global norm 으로 clipping 후, optimizer를 통해 모델 variable들을 업데이트 하도록 wire한다.

    ```python
    grads, variables = zip(*average_gradients(self.model_builder.get_tower_grads()))
    grads, global_norm = tf.clip_by_global_norm(grads, self.options.max_global_norm)
    train_op = self.optimizer.apply_gradients(zip(grads, variables))
    ```
1. [L80-L86](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L80-L86):
로그 파일 정리

    ```python
    if self.options.clear_logs_before_training:
        shutil.rmtree(self.options.log_dir, ignore_errors=True)
    os.makedirs(self.options.log_dir, exist_ok=True)
    self.train_writer = tf.summary.FileWriter(os.path.join(
        self.options.log_dir, "train"), graph=tf.get_default_graph())
    self.val_writer = tf.summary.FileWriter(os.path.join(
        self.options.log_dir, "val"), graph=tf.get_default_graph())
    ```

1. [L87-L106](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L87-L106):
로그에 summary를 출력하기위한 em, f1, highest_f1, loss, gradients 등의 variable 등을 생성

    ```python
    self.em = tf.Variable(name="em",
        initial_value=0, trainable=False, dtype=tf.float32)
    self.f1 = tf.Variable(name="f1",
        initial_value=0, trainable=False, dtype=tf.float32)
    self.highest_f1 = tf.Variable(name="highest_f1",
        initial_value=0, trainable=False, dtype=tf.float32)
    self.extra_save_vars.append(self.highest_f1)
    highest_f1_placeholder = tf.placeholder(tf.float32)
    assign_highest_f1 = tf.assign(self.highest_f1, highest_f1_placeholder)
    em_summary = tf.summary.scalar("exact_match", self.em)
    f1_summary = tf.summary.scalar("f1_score", self.f1)
    for summary in [self.em, self.f1]:
        assignment_dict = {}
        self.summary_assignments[summary] = assignment_dict
        placeholder = tf.placeholder(tf.float32)
        assignment_dict["placeholder"] = placeholder
        assignment_dict["assign_op"] = tf.assign(summary, placeholder)
    loss = self.model_builder.get_loss()
    loss_summary = tf.summary.scalar("loss", loss)
    gradients_summary = tf.summary.scalar("gradients", global_norm)
    ```

1. [L121-L131](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L121-L131):
Tensorflow session에서 현재 iteration 횟수, iterations_per_epoch, learning_rate 등을 가져와 출력한다.

    ```python
    current_iter = int(self.session.run(iteration_num))
    current_highest_f1 = self.session.run(self.highest_f1)
    current_learning_rate = self.session.run(learning_rate)
    total_ds_size = self.sq_dataset.estimate_total_train_ds_size()
    val_ds_size = self.sq_dataset.estimate_total_dev_ds_size()
    iterations_per_epoch = int(total_ds_size / \
        (self.options.batch_size * max(1, self.options.num_gpus)))
    start_time = time.time()
    print("Current iteration: %d, Iters/epoch: %d, Current learning rate: %f"
          % (current_iter, iterations_per_epoch,
              _get_val(current_learning_rate)))
    ```

1. [L136](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L136):
트레이닝 시작, loop이 한번은 iteration 1번을 실행한다

    ```python
    while True:
    ```

1. [L139-L144](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L139-L144):
iteration 한번 실행

    ```python
    _, loss_value, _, loss_summary_value, \
        gradients_summary_value, norm_value = \
        self.session.run([train_op, loss, incr_iter,
            loss_summary, gradients_summary, global_norm], feed_dict=
            get_train_feed_dict(self.sq_dataset,
                self.options, self.model_builder.get_towers()))
    ```

1. [L150-L175](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L150-L175):
각 iteration관련된 로그 및 요약 (iter #, f1, loss, time per iteration 등) 출력후, iter 종료.

    ```python
    print("iteration:", str(i),
          "highest f1: %.4f" % current_highest_f1,
          "learning rate: %.3E" % _get_val(current_learning_rate),
          "loss: %.3E" % _get_val(loss_value),
          "Sec/iter: %.3f" % time_per_iter, 
          "time/epoch", readable_time(time_per_epoch), end="\r")
    if i % self.options.log_every == 0:
        if self.options.log_gradients:
            self.train_writer.add_summary(gradients_summary_value, i)
        if self.options.log_loss:
            self.train_writer.add_summary(loss_summary_value, i)
    if i % self.options.log_valid_every == 0:
        loss_summary_value, gradients_summary_value, loss_value = \
            self.session.run([
                loss_summary, gradients_summary, loss], 
                feed_dict=get_dev_feed_dict(self.sq_dataset,
                    self.options, self.model_builder.get_towers()))
        self.sq_dataset.increment_val_samples_processed(
            self.options.batch_size * num_towers)
        if self.options.log_gradients:
            self.val_writer.add_summary(gradients_summary_value, i)
        if self.options.log_loss:
            self.val_writer.add_summary(loss_summary_value, i)
        print("")
        print("[Validation] iteration:", str(i),
              "loss: %.3E" % _get_val(loss_value))
    ```

1. [L179-L187](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L179-L187):
한 epoch가 끝나면 training set 과 validation set에 대한 model evaluation 실행

    ```python
    em, f1 = evaluate_train_partial(self.session,
        self.model_builder.get_towers(), self.sq_dataset,
        self.options, sample_limit=val_ds_size)
    print("")
    print("[Train] F1", f1, "Em", em)
    val_em, val_f1 = evaluate_dev_partial(self.session,
        self.model_builder.get_towers(), self.sq_dataset,
        self.options, sample_limit=val_ds_size)
    print("[Valid] F1", val_f1, "Em", val_em)
    ```

1. [L190-L203](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L190-L203):
evaluation 관련 로그 출력

    ```python
    self._perform_summary_assignment(self.em, em)
    self._perform_summary_assignment(self.f1, f1)
    if self.options.log_exact_match:
        self.train_writer.add_summary(self.session.run(em_summary), i)
    if self.options.log_f1_score:
        self.train_writer.add_summary(self.session.run(f1_summary), i)
    self._perform_summary_assignment(self.em, val_em)
    self._perform_summary_assignment(self.f1, val_f1)
    if self.options.log_exact_match:
        self.val_writer.add_summary(self.session.run(em_summary), i)
    if self.options.log_f1_score:
        self.val_writer.add_summary(self.session.run(f1_summary), i)
    self.train_writer.flush()
    self.val_writer.flush()
    ```

1. [L204-L222](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L204-L222):
evaluation 결과에서 모델이 bad_checkpoints_tolerance 만큼 향상되지 않았을경우, learning_rate를 더 작게 조정한다.

    ```python
    # If the validation F1 score didn't increase, then cut
    # the learning rate.
    if current_highest_f1 >= val_f1:
        if num_bad_checkpoints \
            < self.options.bad_checkpoints_tolerance:
            num_bad_checkpoints += 1
            print("Hit bad checkpoint. num_bad_checkpoints: %d"
                % num_bad_checkpoints)
        else:
            new_learning_rate = current_learning_rate \
                * self.options.bad_iteration_learning_decay
            self.session.run(assign_learning_rate, feed_dict={
                learning_rate_placeholder: new_learning_rate})
            current_learning_rate = new_learning_rate
            print(("Dropped learning rate to %.3E because val F1 "
                + "didn't increase from %.3E")
                % (_get_val(new_learning_rate),
                   _get_val(current_highest_f1)))
            num_bad_checkpoints = 0
    ```

1. [L224-L232](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L224-L232):
evaluation 결과에서 모델이 향상되었을 경우 모델을 저장하고 향상된 F1을 출력한다.

    ```python
    self.session.run(assign_highest_f1, feed_dict={
        highest_f1_placeholder: val_f1})
    current_highest_f1 = val_f1
    print("Achieved new highest F1: %f" % val_f1)
    self.saver.save(self.session, self.checkpoint_file_name)
    maybe_upload_files_to_s3(self.s3, self.s3_save_key,
        self.options.checkpoint_dir, self.options)
    print("Saved model at iteration", i)
    num_bad_checkpoints = 0
    ```

1. [L233-L237](https://github.com/DeepLearningCollege/intQA/blob/2418ecc92c80c9bad93835e4d3983605c9bb2856/train/trainer.py#L233-L237):
epoch 종료

    ```python
    print("Total epoch time %s" % readable_time(
        time.time() - epoch_start))
    print("Cumulative run time %s" % readable_time(
        time.time() - train_start_time))
    epoch_start = time.time()
    ```