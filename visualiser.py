import matplotlib.pyplot as plt
import numpy as np

def visualise_image(sess, mnist, x_, y_, y, dim_x=28, dim_y=28, index=0):
    output = sess.run(y, feed_dict={x_: mnist.test.images[index:index+1], y_: mnist.test.images[index:index+1]})
    input_img = np.reshape(mnist.test.images[index], (dim_x,dim_y))
    output_img = np.reshape(output, (dim_x, dim_y))
    plot_image = np.concatenate((output_img, input_img), axis=1)
    plt.imshow(plot_image, cmap='gray')
    plt.show()

def visualise_keras(model, input_data, dim_x=28, dim_y=28, extra_dim=None, transform_required=False):
    output_data = model.predict(input_data[:3])
    reshape_dim = (dim_x, dim_y, extra_dim) if extra_dim is not None else (dim_x, dim_y)
    for i in range(3):
        input_img = np.reshape(input_data[i], reshape_dim) if transform_required else input_data[i]
        output_img = np.reshape(output_data[i], reshape_dim)
        plt.subplot(321 + i*2)
        plt.imshow(input_img)
        plt.subplot(321+ i*2 + 1)
        plt.imshow(output_img)
    plt.show()

def visualise_raw(input_data, transform_required=False, dim_x=28, dim_y=28):
    for i in range(8):
        if transform_required:
            input_img = np.reshape(input_data[i], (dim_x, dim_y))
        else:
            input_img = input_data[i]
        plt.subplot(421 + i)
        plt.imshow(input_img, cmap='gray')
    plt.show()