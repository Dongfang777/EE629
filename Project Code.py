import sys
import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
import random
import sys
from sklearn.model_selection import train_test_split

def save_img():
    video_path = './data/sc/'
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = video_path + file_name
        os.makedirs(folder_name,exist_ok=True)
        vc = cv2.VideoCapture(video_path+video_name) 
        c=0
        rval=vc.isOpened()

        while rval:  
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name+'/'
            if rval:
                cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame) 
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)
save_img()


input_dir = './data/sc/1'
output_dir = './data/my_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_img, 1)


            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)

input_dir = './data/sc/2'
output_dir = './data/other_faces'
size = 64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        
        if filename.endswith('.png'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename

            img = cv2.imread(img_path)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dets = detector(gray_img, 1)


            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]

                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)

                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)


my_faces_path = './data/my_faces'
other_faces_path = './data/other_faces'
size = 64

imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)

imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])


train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.2, random_state=random.randint(0,100))

train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    W1 = weightVariable([3,3,3,32]) 
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)


    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)


    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)


    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    tf.add_to_collection('pred_network', out)
    return out


out = cnnLayer()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

sess  = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

for n in range(10):
           
    for i in range(num_batch):
        batch_x = train_x[i*batch_size : (i+1)*batch_size]
        batch_y = train_y[i*batch_size : (i+1)*batch_size]
                
        _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
        summary_writer.add_summary(summary, n*num_batch+i)
        print("loss",n*num_batch+i, loss)

        if (n*num_batch+i) % 30 == 0:
            acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
            print("acc",n*num_batch+i, acc)

            if acc > 0.98 and n > 2:
                sys.exit(0)
    print('accuracy less 0.98, exited!')
    
    
    def is_my_face(image,sess,out):  
    pre = tf.nn.softmax(out)
#     sess.run(pre,feed_dict={x:test_x, keep_prob_5:0.5,keep_prob_75:0.75})
    res = sess.run(pre, feed_dict={x: image.astype('float32')/255.0, keep_prob_5:1.0, keep_prob_75: 1.0})  
    print(res)
    if np.argmax(res[0]) == 1:  
        return True  
    else:  
        return False  

def test(path,sess,out):
         
    detector = dlib.get_frontal_face_detector()

    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_img, 1)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        face = cv2.resize(face, (size,size))
        cv2.imwrite('./data/temp/1.jpg', face)
        
        temps=[]
        temp = cv2.imread('./data/temp/1.jpg')
        top,bottom,left,right = getPaddingSize(temp)
        temp = cv2.copyMakeBorder(temp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        temp = cv2.resize(temp, (64, 64))
        temps.append(temp)
        
        temps=np.array(temps)
         
        X=temps.reshape(temps.shape[0], 64, 64, 3)
        
        
        flag=is_my_face(X,sess,out)
        if flag:
            print('me!')
        else:
            print('others!')

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)


# Run a test on my code:
# #测试本人
# img_path='./data/sc/1/1_504.jpg'

# #测试其他人
# img_path='./data/sc/2/00890.png'

test(img_path,sess,out)
