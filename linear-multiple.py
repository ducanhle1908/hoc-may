# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

filename = 'kc_house_data.csv'
#load csv data
puredata = np.loadtxt(filename, delimiter=',')
X = puredata[:,1:]
Y = puredata[:,0]

#chuẩn hóa thuộc tính để thuật toán hội tụ nhanh hơn
def normalization(x):
    mean_x = [];                                             #Khởi tạo các ma trận để chứa các giá trị trung bình và độ lệch chuẩn sau khi tính toán
    std_x = [];
    X_normalized = x;                                        #Gán X_normalized bằng ma trận của tập dữ liệu huấn luyện
    temp = x.shape[1]
    for i in range(temp):
        m = np.mean(x[:, i])                                 #Tính giá trị trung bình của mỗi thuộc tính trong tập huấn luyện
        s = np.std(x[:, i])                                  #Độ lệch chuẩn
        mean_x.append(m)                                     #Thêm các giá trị vừa tìm được vào ma trận ban đầu
        std_x.append(s)
        X_normalized[:, i] = (X_normalized[:, i] - m) / s    #Trừ đi giá trị trung bình của mỗi thuộc tính trong tập huấn luyện         #Sau đó chia cho độ lệch chuẩn tương ứng của mỗi thuộc tính
    return X_normalized, mean_x, std_x

def cost(x,y,theta):                                         #Hàm mất mát J(theta)
    m = y.size                                               #số lượng mẫu training
    predicted = np.dot(x,theta)                              
    sqErr = (predicted - y)
    J = ((1.0) / (2 * m)) * np.dot(sqErr.T, sqErr)**2
    return J

def gradient_descent(x, y, theta, alpha, iterations):                                #thuật toán gradient descent tìm giá trị tối ưu cho theta                                                       
    m = y.size                                              
    theta_n = theta.size                                                              #theta size

    J_theta_log = np.zeros(shape=(iterations+1, 1))                                  #tạo 1 ma trận để lưu trữ các giá trị trả về trong gradient descent  
    J_theta_log[0, 0] = cost(x, y, theta)                                             #(Biến tạm để kiểm tra tiến độ của gradient descent)
 
    for i in range(iterations):                                                        #Vòng lặp cho gradient descent
        predicted = x.dot(theta)

        for thetas in range(theta_n):
            tmp = x[:,thetas]
            tmp.shape = (m,1)
            err = (predicted - y) * tmp                                              #tính sai số (predict – y)
            theta[thetas][0] = theta[thetas][0] - alpha * (1.0 / m) * err.sum()      #thực hiện gradient descent để cập nhật theta
        J_theta_log[i+1, 0] = cost(x, y, theta)

    return theta, J_theta_log

#kích thước của tập huấn luyện
m,n = np.shape(X)
#format Y thành ma trận [m,1]
Y.shape = (m, 1)
#Scale features
x_scale, mean_r, std_r = normalization(X)

#Thêm vào cột đầu tiên của ma trận X giá trị bằng 1 
XX = np.ones(shape=(m,1))
XX = np.append(XX,x_scale,1)

#khởi tạo giá trị ban đầu cho thetas bằng 0
theta = np.zeros(shape=(n+1, 1))
#Thiết lập số vòng lặp và learning_rate
iterations = 1000
alpha = 0.09
#tính theta bằng thuật toán gradient descent
theta, J_theta_log = gradient_descent(XX, Y, theta, alpha, iterations)
print("Theta:")
print(theta)

home_price=np.array([1.0,
                     (5.0 - mean_r[0])/std_r[0],
                     (2.75 - mean_r[1])/std_r[1],
                     (3078.0 - mean_r[2])/std_r[2],
                     (6371.0 - mean_r[3])/std_r[3],
                     (2.0 - mean_r[4])/std_r[4],
                     (0.0 - mean_r[5])/std_r[5],
                     (0.0 - mean_r[6])/std_r[6],
                     (3.0 - mean_r[7])/std_r[7],
                     (9.0 - mean_r[8])/std_r[8],
                     (3078.0 - mean_r[9])/std_r[9],
                     (0.0 - mean_r[10])/std_r[10],
                     (2014.0 - mean_r[11])/std_r[11],
                     (0.0 - mean_r[12])/std_r[12],
                     (98042.0 - mean_r[13])/std_r[13],
                     (47.3587 - mean_r[14])/std_r[14],
                     (-121.163 - mean_r[15])/std_r[15],
                     (1979.0 - mean_r[16])/std_r[16],
                     (19030.0 - mean_r[17])/std_r[17]
                     ]).dot(theta)
print ('Home price: %f' % (home_price))

# # predict = predict(X, theta)
# plt.figure(1)
# plt.plot(X[:,1],Y,'rx')
# # plt.plot(X[:,1],predict,'-b') 
# plt.show()

#Biểu diễn hàm mất mát bằng đồ thị
fig = plt.figure('Cost function convergence')
plt.plot(J_theta_log)
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function convergence')
plt.show()

#Dự đoán



