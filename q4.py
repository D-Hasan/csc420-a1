import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def gaussian_kernel(std):
    '''
    Creates a 2D gaussian kernel, given a standard deviation.

    std: standard deviation of gaussian (isotropic)
    '''

    gaussian_pdf = lambda x: 1/(np.sqrt(2*np.pi * std**2)) * np.exp(-x**2/(std**2 * 2))

    num_stds = 3
    dimension = int(2*(np.ceil(std)*num_stds) + 1)
    center = std *num_stds
    gaussian_1D = np.zeros((dimension, 1))

    for i in range(dimension):
        p_x = gaussian_pdf(i-center)
        gaussian_1D[i] = p_x 

    gaussian_2D = gaussian_1D @ gaussian_1D.T
    gaussian_2D /= gaussian_2D.sum() 

    return gaussian_2D

def display_image(img, cmap=None):
    '''
    Plots image.
    '''
    plt.cla()

    if cmap:
        plt.imshow(img, cmap='gray')
    else: 
        plt.imshow(img)

def convolution(img, filter):
    '''
    Convolves an image (h x w) with a filter (k x k).

    img: array of (h x w x channels)
    filter: array of (k x k)
    '''
    h, w, channels = img.shape 
    k = filter.shape[0]

    padding = int((k-1)/2)
    padded_image = np.zeros((h + padding*2, w + padding*2, channels), int)
    padded_image[padding:-padding, padding:-padding, :] = img 

    output = np.zeros((h, w, channels), int)
    filter = filter[..., np.newaxis]

    # just to be consistent
    filter = np.flip(filter, 0)
    filter = np.flip(filter, 1)

    for x in range(h):
        for y in range(w):
            output[x, y, :] = (filter * padded_image[x: x+k, y:y+k]).sum(axis=(0, 1)).astype(int)

    return output

def sobel_kernels():
    '''
    Returns Sobel kernels for calculating image gradient for dx and dy.
    '''
    x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ], int)
    y_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
        ], int)

    
    return x_kernel, y_kernel

def get_grad_magnitude(img):
    '''
    Computes gradient of image and returns gradient magnitude image.

    img: array of (h x w x channels)
    '''
    img_gray = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    sobel_x, sobel_y = sobel_kernels()
    
    grad_x = convolution(img_gray, sobel_x)
    grad_y = convolution(img_gray, sobel_y)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    return grad_mag 

def threshold(gradient_img, eps=3):
    t_curr = gradient_img.mean()
    t_prev = t_curr + eps + 1

    i = 0
    while abs(t_curr - t_prev) > eps:
        lower = gradient_img[gradient_img < t_curr]
        upper = gradient_img[gradient_img >= t_curr]

        mean_lower = lower.mean()
        mean_upper = upper.mean()

        t_prev = t_curr
        t_curr = (mean_lower + mean_upper)/2
        i += 1


    edge = np.zeros(gradient_img.shape, int)
    edge[gradient_img >= t_curr] = 255 

    return edge 

def q4_end_to_end(path, name, gaussian_k=3, eps=3):
    print('Q4 Edge Detection for ', path)
    img = cv2.imread(path)[..., ::-1]
    
    blurred = convolution(img, gaussian_kernel(gaussian_k))
    display_image(blurred)
    plt.savefig('images/q4_blurred_' + name + '.png')

    grad_img = get_grad_magnitude(blurred)
    display_image(grad_img, 'gray')
    plt.savefig('images/q4_gradient_' + name + '.png')

    output = threshold(grad_img, eps)
    display_image(output, 'gray')
    plt.savefig('images/q4_edges_' + name + '.png')

    return img, blurred, grad_img, output 

def q4_step1():
    print('Q4 Step 1, visualizing gaussian kernels')
    img = cv2.imread('Q4_image_1.jpg')[..., ::-1]

    gaussian_std_1 = gaussian_kernel(1)
    gaussian_std_3 = gaussian_kernel(3)

    blurred_std_1 = convolution(img, gaussian_std_1)
    blurred_std_3 = convolution(img, gaussian_std_3)

    figure, axs = plt.subplots(2, 2)
    axs[0][0].imshow(gaussian_std_1, cmap='gray')
    axs[0][0].set_title('Gaussian Kernel with std=1')

    axs[0][1].imshow(gaussian_std_3, cmap='gray')
    axs[0][1].set_title('Gaussian Kernel with std=3')

    axs[1][0].imshow(blurred_std_1)
    axs[1][1].imshow(blurred_std_3)

    plt.savefig('images/q4_step1.png')
    plt.clf(); 


def testing(path):
    img = cv2.imread(path)[..., ::-1]

    blurred = cv2.GaussianBlur(img,(19,19),3)
    blurred_gray = cv2.cvtColor(np.float32(blurred), cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(blurred_gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blurred_gray,cv2.CV_64F,0,1,ksize=3)

    grad_img = np.sqrt(sobelx**2 + sobely**2)

    output = threshold(grad_img, 3)
    display_image(output, 'gray')
    return img, blurred, grad_img, output 


if __name__ == '__main__':
    q4_step1()

    q4_end_to_end('Q4_image_1.jpg', 'img1')
    q4_end_to_end('Q4_image_2.jpg', 'img2')
    q4_end_to_end('mine.jpg', 'mine')



