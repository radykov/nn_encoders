import matplotlib.pyplot as plt
import numpy as np

def visualise_image(sess, mnist, x_, y_, y, index=0):
    output = sess.run(y, feed_dict={x_: mnist.test.images[index:index+1], y_: mnist.test.images[index:index+1]})
    input_img = np.reshape(mnist.test.images[index], (28,28))
    output_img = np.reshape(output, (28, 28))
    plot_image = np.concatenate((output_img, input_img), axis=1)
    plt.imshow(plot_image, cmap='gray')
    plt.show()

def visualise_keras(model, input_data):
    output_data = model.predict(input_data[1000:1003])
    for i in range(3):
        input_img = np.reshape(input_data[i+1000], (28, 28))
        output_img = np.reshape(output_data[i], (28, 28))
        plt.subplot(321 + i*2)
        plt.imshow(input_img, cmap='gray')
        plt.subplot(321+ i*2 + 1)
        plt.imshow(output_img, cmap='gray')
    plt.show()