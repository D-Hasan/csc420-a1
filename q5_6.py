import numpy as np 

import q4

def connected_components(edge_image):
    '''
    Returns labelled image of componenets
    '''

    labels = np.zeros(edge_image.shape, int)

    curr_label = 0

    for x in range(edge_image.shape[0]):
        for y in range(edge_image.shape[1]):
            queue = []
            curr_label += 1

            pixel = edge_image[x, y, 0]
            label = labels[x, y, 0]

            if pixel > 0 and label == 0:
                labels[x][y][0] = curr_label
                queue.append((x, y))
            else:
                continue 

            while len(queue) > 0:
                x, y = queue.pop(0)
                
                neighbour_coords = [-1, 0, 1]
                for i in neighbour_coords:
                    for j in neighbour_coords:
                        if (x+i >= 0 and y+j>= 0) and (x+i < labels.shape[0] and y+j < labels.shape[1]):
                            neighbour_pixel = edge_image[x+i, y+j, 0]
                            neighbour_label = labels[x+i, y+j, 0]

                            if neighbour_pixel > 0 and neighbour_label == 0:
                                labels[x+i][y+j] = curr_label 
                                queue.append((x+i, y+j))
                            else:
                                continue 
    return labels 
    


def q5_end_to_end(path, sigma=1, eps=3):
    img, blurred, grad_img, output = q4.q4_end_to_end(path, 'q5_std' + str(sigma), sigma, eps)
    labels = connected_components(output)
    return print('Num of Components: {}, sigma = {}, eps = {}'.format(len(np.unique(labels)), sigma, eps))

if __name__ == '__main__':
    q5_end_to_end('Q6.png', sigma=1, eps=3)
    q5_end_to_end('Q6.png', sigma=2, eps=3)
    q5_end_to_end('Q6.png', sigma=3, eps=3)