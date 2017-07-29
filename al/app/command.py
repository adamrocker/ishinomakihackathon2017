import tensorflow as tf

tf.app.flags.DEFINE_boolean("bool", True, "bool value")
tf.app.flags.DEFINE_integer("int", 0, "int value")
tf.app.flags.DEFINE_string("str", "str", "string value")
tf.app.flags.DEFINE_string("test_str", "test", "test string value")

def main(argv):
  flags = tf.app.flags.FLAGS

  print(flags.bool, flags.int, flags.str, flags.test_str)


if __name__ == '__main__':
    tf.app.run()
