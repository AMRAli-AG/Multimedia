# ----------------------------------------
# الجزء الأول: معالجة الصور
# ----------------------------------------

import cv2  # مكتبة OpenCV لمعالجة الصور
from google.colab.patches import cv2_imshow  # لعرض الصور في Google Colab
import numpy as np  # مكتبة NumPy للتعامل مع المصفوفات

# قراءة الصورة وعرضها
img = cv2.imread("peppers.png", 1)  # 1: قراءة الصورة بالألوان (لو كان 0 يبقى تدرج رمادي)
print("أبعاد الصورة:", img.shape)  # طباعة أبعاد الصورة (ارتفاع, عرض, عدد القنوات)

# إزالة القنوات الزرقاء والخضراء (إبقاء القناة الحمراء فقط)
img[:, :, 0] = 0  # 0: القناة الزرقاء (إزالة اللون الأزرق)
img[:, :, 1] = 0  # 1: القناة الخضراء (إزالة اللون الأخضر)

# عرض الصورة المعدلة
print("الصورة بعد إزالة الأزرق والأخضر:")
cv2_imshow(img)

# ----------------------------------------
# الجزء الثاني: استخدام الفلاتر
# ----------------------------------------

# قراءة صورة أخرى
img = cv2.imread('cameraman.tif')  # قراءة صورة "cameraman.tif"
print("الصورة الأصلية:")
cv2_imshow(img)

# تطبيق فلتر متوسط
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  # 9: مجموع قيم النواة (متوسط القيم)
img_f = cv2.filter2D(img, -1, kernel)  # تطبيق الفلتر
print("الصورة بعد الفلتر المتوسط:")
cv2_imshow(img_f)

# تطبيق فلتر آخر
kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]) / 15  # 15: مجموع قيم النواة (مرجحة)
img_f = cv2.filter2D(img, -1, kernel)  # تطبيق الفلتر
print("الصورة بعد الفلتر المرجح:")
cv2_imshow(img_f)

# تطبيق Gaussian Blur
img_f = cv2.GaussianBlur(img, (11, 11), 2)  # (11, 11): حجم النواة، 2: قيمة سيجما (مدى التمويه)
print("الصورة بعد Gaussian Blur (سيجما = 2):")
cv2_imshow(img_f)

img_f = cv2.GaussianBlur(img, (11, 11), 5)  # 5: قيمة سيجما أكبر (تمويه أقوى)
print("الصورة بعد Gaussian Blur (سيجما = 5):")
cv2_imshow(img_f)

# ----------------------------------------
# الجزء الثالث: اكتشاف الحواف
# ----------------------------------------

# تطبيق فلتر Sobel
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # نواة للاتجاه الأفقي
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # نواة للاتجاه الرأسي

img_x = cv2.filter2D(img, -1, kernel_x)  # تطبيق الفلتر الأفقي
img_y = cv2.filter2D(img, -1, kernel_y)  # تطبيق الفلتر الرأسي

print("الصورة الأصلية:")
cv2_imshow(img)
print("الصورة بعد فلتر Sobel الأفقي:")
cv2_imshow(img_x)
print("الصورة بعد فلتر Sobel الرأسي:")
cv2_imshow(img_y)

# تطبيق فلتر Laplacian
kernel_x = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # نواة Laplacian
img_f = cv2.filter2D(img, -1, kernel_x)  # تطبيق الفلتر
print("الصورة بعد فلتر Laplacian:")
cv2_imshow(img_f)

# تطبيق Canny Edge Detection
img_f = cv2.Canny(img, 150, 240)  # 150: الحد الأدنى للحواف، 240: الحد الأعلى للحواف
print("الصورة الأصلية:")
cv2_imshow(img)
print("الصورة بعد اكتشاف الحواف باستخدام Canny:")
cv2_imshow(img_f)

# ----------------------------------------
# الجزء الرابع: الفرق بين صورتين بعد التمويه
# ----------------------------------------

# قراءة الصورة بتدرج رمادي
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)  # قراءة الصورة بتدرج رمادي

# تطبيق Gaussian Blur بقيم سيجما مختلفة
sigma1 = 1  # 1: قيمة سيجما الأولى (تمويه خفيف)
blur1 = cv2.GaussianBlur(img, (11, 11), sigma1)  # تطبيق التمويه

sigma2 = 5  # 5: قيمة سيجما الثانية (تمويه أقوى)
blur2 = cv2.GaussianBlur(img, (11, 11), sigma2)  # تطبيق التمويه

# حساب الفرق بين الصورتين
diff_img = cv2.subtract(blur1, blur2)  # طرح الصورتين للحصول على الفرق

# عرض الصور
print("الصورة الأصلية:")
cv2_imshow(img)
print("الصورة بعد Gaussian Blur (سيجما = 1):")
cv2_imshow(blur1)
print("الصورة بعد Gaussian Blur (سيجما = 5):")
cv2_imshow(blur2)
print("الفرق بين الصورتين:")
cv2_imshow(diff_img)

# ----------------------------------------
# الجزء الخامس: تصنيف الصور باستخدام k-Nearest Neighbors (kNN)
# ----------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# تحميل بيانات CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# تقسيم البيانات إلى تدريب واختبار
X_train, X_val, y_train, y_val = train_test_split(X_train[0:5000], y_train[0:5000], test_size=0.1, random_state=42)
# 5000: عدد الصور المستخدمة، 0.1: نسبة الاختبار (10%)

# تسطيح الصور وتطبيعها
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 255: تطبيع القيم بين 0 و1
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# إنشاء وتدريب نموذج kNN
knn = KNeighborsClassifier(n_neighbors=5)  # 5: عدد الجيران المستخدمة في النموذج
knn.fit(X_train, y_train)  # تدريب النموذج

# التنبؤ على بيانات الاختبار
y_val_pred = knn.predict(X_val)  # التنبؤ بتسميات بيانات الاختبار
accuracy_test = accuracy_score(y_val, y_val_pred)  # حساب الدقة
print("دقة النموذج على بيانات الاختبار:", accuracy_test)

# ----------------------------------------
# الجزء السادس: تصنيف الصور باستخدام MLP و ConvNet
# ----------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# تحميل بيانات CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# تطبيع القيم بين 0 و1
x_train = x_train.astype('float32') / 255.0  # 255: تطبيع القيم بين 0 و1
x_test = x_test.astype('float32') / 255.0

# بناء نموذج ConvNet
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 32: عدد الفلاتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64: عدد الفلاتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128: عدد الفلاتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Flatten())  # تسطيح البيانات
model.add(Dense(128, activation='relu'))  # 128: عدد العصبونات في الطبقة الكثيفة
model.add(Dense(10, activation='softmax'))  # 10: عدد العصبونات في الطبقة الأخيرة

# تجميع النموذج
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))  # 10: عدد الدورات التدريبية

# تقييم النموذج
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("دقة النموذج على بيانات الاختبار:", test_acc)
