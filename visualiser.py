import matplotlib.pyplot as plt
import numpy as np

def visualise_image(sess, mnist, x_, y_, y, index=0):
    output = sess.run(y, feed_dict={x_: mnist.test.images[index:index+1], y_: mnist.test.images[index:index+1]})
    input_img = np.reshape(mnist.test.images[index], (28,28))
    output_img = np.reshape(output, (28, 28))
    plot_image = np.concatenate((output_img, input_img), axis=1)
    plt.imshow(plot_image, cmap='gray')
    # plt.imshow(input_img, cmap='gray')
    # plt.imshow(output_img, cmap='gray')
    plt.show()