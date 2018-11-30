import cv2

def image_enhancement(img_norm, img):
    for i in range(int(img_norm.shape[0]/32)):
        for j in range(int(img_norm.shape[1]/32)):
            block = img_norm[i*32:(i+1)*32, j*32:(j+1)*32]
            block = block.astype('uint8')
            temp = cv2.equalizeHist(block)
            img_norm[i*32:(i+1)*32, j*32:(j+1)*32] = temp        
    img_enhance = img_norm[:48]
    return img_enhance