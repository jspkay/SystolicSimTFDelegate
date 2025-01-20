import tensorflow as tf 
import sys
import numpy as np

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: tryloading.py path/to/delegate.so [networkFileName]")
    exit(-1)

delegatePath = sys.argv[1]

delegate = tf.lite.experimental.load_delegate(
    delegatePath,
    options={
        "print_log_to_stdout": "true",
    },
)

# print(delegate)

networkName = "mnist.tflite"
# "AlexNet-quantized.tflite",

if len(sys.argv) == 3:
    networkName = sys.argv[2]

intr = tf.lite.Interpreter(
    networkName,
    experimental_delegates = [delegate]
)
intr.allocate_tensors()

print(intr)

input_details = intr.get_input_details()
output_details = intr.get_output_details()

input_shape = input_details[0]["shape"]
print(f"Input shape is {input_shape}")

# data = 32) #  np.random.random_sample(input_shape).astype(np.float32)
a = b"\xde\xad\xbe\xef" # deadbeef signature
# data = np.frombuffer(a, np.float32) * np.ones(input_shape)
data = 123.1 * np.ones(input_shape)
intr.set_tensor(input_details[0]["index"], data.astype(np.float32))

# print(f"PYTHON: data(deadbeef)  with shape {input_shape} is {data}")
intr.invoke()

output_data = intr.get_tensor(output_details[0]['index'])
print(output_data)

