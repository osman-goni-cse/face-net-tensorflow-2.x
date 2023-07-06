# import tensorflow as tf
# import align.detect_face

# tf.compat.v1.disable_eager_execution()
 
# sess=tf.compat.v1.Session()    
# #First let's load meta graph and restore weights
# saver = tf.compat.v1.train.import_meta_graph('/home/frt/ConvertedToTf2/models/facenet/20230621-183750/model-20230621-183750.meta')
# saver.restore(sess,tf.train.latest_checkpoint('/home/frt/ConvertedToTf2/models/facenet/20230621-183750'))

# graph = tf.compat.v1.get_default_graph()
# input_tensor = graph.get_tensor_by_name('input:0')
# embeddings = graph.get_tensor_by_name('embeddings:0')
# phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

# print(input_tensor, embeddings)
# # output_tensor = graph.get_tensor_by_name('output_tensor_name:0')

# # # Perform inference on new data
# import numpy as np
# from PIL import Image

# # Load the saved model

# # Set the image path
# image_path = '/home/frt/ConvertedToTf2/dsi_employee'

# for i in range(nrof_samples):
#     image_path = os.path.join(folder_path, image_paths[i])
#     img = imageio.imread(image_path)
#     img_size = img.shape[:2]
    
#     # Perform face detection and alignment
#     bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#     count_per_image.append(len(bounding_boxes))
    
#     for j in range(len(bounding_boxes)):
#         det = np.squeeze(bounding_boxes[j, 0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        
#         cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#         aligned = imageio.imresize(cropped, (image_size, image_size), interp='bilinear')
#         prewhitened = facenet.prewhiten(aligned)
        
#         img_list.append(prewhitened)

# # Stack the preprocessed images into a numpy array
# images = np.stack(img_list)

# # Load and preprocess the image
# image = Image.open(image_path)
# image = image.convert('RGB') 
# image = image.resize((160, 160))  # Resize the image to match the model's input size
# image = np.array(image) / 255.0  # Normalize the image

# # Perform prediction
# new_data = np.expand_dims(image, axis=0)

# feed_dict = { input_tensor: new_data , phase_train_placeholder:False}
# emb = sess.run(embeddings, feed_dict=feed_dict)
# print(emb)

import joblib

classifier_filename_exp = 'osman/svm.pkl'

try:
    model = joblib.load(classifier_filename_exp)
    class_names = None  # Set class_names to None or provide the appropriate default value

    print(model)
    print(class_names)
    print("Success")
except FileNotFoundError:
    print(f"Error: File {classifier_filename_exp} not found.")
except Exception as e:
    print(f"Error: Failed to load model from file {classifier_filename_exp}.")
    print(e)
