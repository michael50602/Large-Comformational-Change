import tensorflow as tf
def tf_confusion_matrix(pred, actuals):
  predictions = tf.argmax(pred, 1)
  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      tf.float32
    )
  )

  tn = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      tf.float32
    )
  )

  fp = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      tf.float32
    )
  )

  fn = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      tf.float32
    )
  )

  accuracy = tf.truediv(tf.add(tp, tn), tf.add(tf.add(tp,fp),tf.add(fn, tn)))
  recall = tf.truediv(tp, tf.add(tp, fn))
  precision = tf.truediv(tp, tf.add(tp, fp))
  
  f1_score = (2 * (precision * recall)) / (precision + recall)
  
  return accuracy, recall, precision, f1_score
