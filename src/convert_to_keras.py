import tensorflow as tf

# Load your existing .h5 model
model = tf.keras.models.load_model('final_resnet50_deepfake.h5')

# Save it in the new .keras format
model.save('final_resnet50_deepfake.keras')
