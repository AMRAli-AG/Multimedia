# ----------------------------------------
# الجزء الأول: معالجة الصور
# ----------------------------------------

import cv2  # مكتبة OpenCV علشان نتعامل مع الصور
from google.colab.patches import cv2_imshow  # علشان نعرض الصور في Google Colab
import numpy as np  # مكتبة NumPy علشان نتعامل مع المصفوفات

# قراءة الصورة وعرضها
img = cv2.imread("peppers.png", 1)  # 1: بنقرأ الصورة بالألوان (لو كان 0 يبقى تدرج رمادي)
# "peppers.png": اسم الصورة اللي بنقرأها
print("أبعاد الصورة:", img.shape)  # بنطبع أبعاد الصورة (الارتفاع، العرض، عدد القنوات)
# img.shape: بيطبع أبعاد الصورة (مثلاً: (300, 400, 3))

# إزالة القنوات الزرقاء والخضراء (إبقاء القناة الحمراء فقط)
img[:, :, 0] = 0  # 0: القناة الزرقاء (بنشيل اللون الأزرق)
# img[:, :, 0]: كل البكسلات في القناة الزرقاء (القناة الأولى)
img[:, :, 1] = 0  # 1: القناة الخضراء (بنشيل اللون الأخضر)
# img[:, :, 1]: كل البكسلات في القناة الخضراء (القناة الثانية)

# عرض الصورة المعدلة
print("الصورة بعد إزالة الأزرق والأخضر:")
cv2_imshow(img)  # بنعرض الصورة في Google Colab

# ----------------------------------------
# الجزء الثاني: استخدام الفلاتر
# ----------------------------------------

# قراءة صورة تانيه
img = cv2.imread('cameraman.tif')  # بنقرأ صورة "cameraman.tif"
# 'cameraman.tif': اسم الصورة
print("الصورة الأصلية:")
cv2_imshow(img)  # بنعرض الصورة الأصلية

# تطبيق فلتر متوسط
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  # 9: مجموع قيم النواة (متوسط القيم)
# kernel: النواة بتاعة الفلتر (مصفوفة 3x3)، 9: مجموع القيم علشان نحسب المتوسط
img_f = cv2.filter2D(img, -1, kernel)  # بنطبق الفلتر على الصورة
# cv2.filter2D: بنستخدم النواة علشان نطبق الفلتر
print("الصورة بعد الفلتر المتوسط:")
cv2_imshow(img_f)  # بنعرض الصورة بعد الفلتر

# تطبيق فلتر تاني
kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]) / 15  # 15: مجموع قيم النواة (مرجحة)
# kernel: النواة بتاعة الفلتر (مصفوفة 3x3)، 15: مجموع القيم علشان نحسب المتوسط المرجح
img_f = cv2.filter2D(img, -1, kernel)  # بنطبق الفلتر
print("الصورة بعد الفلتر المرجح:")
cv2_imshow(img_f)  # بنعرض الصورة بعد الفلتر

# تطبيق Gaussian Blur
img_f = cv2.GaussianBlur(img, (11, 11), 2)  # (11, 11): حجم النواة، 2: قيمة سيجما (مدى التمويه)
# (11, 11): حجم النواة، 2: قيمة سيجما (مدى التمويه)
print("الصورة بعد Gaussian Blur (سيجما = 2):")
cv2_imshow(img_f)  # بنعرض الصورة بعد التمويه

img_f = cv2.GaussianBlur(img, (11, 11), 5)  # 5: قيمة سيجما أكبر (تمويه أقوى)
# 5: قيمة سيجما أكبر (تمويه أقوى)
print("الصورة بعد Gaussian Blur (سيجما = 5):")
cv2_imshow(img_f)  # بنعرض الصورة بعد التمويه

# ----------------------------------------
# الجزء الثالث: اكتشاف الحواف
# ----------------------------------------

# تطبيق فلتر Sobel
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # نواة للاتجاه الأفقي
# kernel_x: نواة Sobel علشان نكتشف الحواف في الاتجاه الأفقي
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # نواة للاتجاه الرأسي
# kernel_y: نواة Sobel علشان نكتشف الحواف في الاتجاه الرأسي

img_x = cv2.filter2D(img, -1, kernel_x)  # بنطبق الفلتر الأفقي
img_y = cv2.filter2D(img, -1, kernel_y)  # بنطبق الفلتر الرأسي

print("الصورة الأصلية:")
cv2_imshow(img)
print("الصورة بعد فلتر Sobel الأفقي:")
cv2_imshow(img_x)
print("الصورة بعد فلتر Sobel الرأسي:")
cv2_imshow(img_y)

# تطبيق فلتر Laplacian
kernel_x = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # نواة Laplacian
# kernel_x: نواة Laplacian علشان نكتشف الحواف
img_f = cv2.filter2D(img, -1, kernel_x)  # بنطبق الفلتر
print("الصورة بعد فلتر Laplacian:")
cv2_imshow(img_f)

# تطبيق Canny Edge Detection
img_f = cv2.Canny(img, 150, 240)  # 150: الحد الأدنى للحواف، 240: الحد الأعلى للحواف
# 150: الحد الأدنى علشان نحدد الحواف، 240: الحد الأعلى
print("الصورة الأصلية:")
cv2_imshow(img)
print("الصورة بعد اكتشاف الحواف باستخدام Canny:")
cv2_imshow(img_f)

# ----------------------------------------
# الجزء الرابع: الفرق بين صورتين بعد التمويه
# ----------------------------------------

# قراءة الصورة بتدرج رمادي
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)  # بنقرأ الصورة بتدرج رمادي
# cv2.IMREAD_GRAYSCALE: علشان نقرأ الصورة بتدرج رمادي

# تطبيق Gaussian Blur بقيم سيجما مختلفة
sigma1 = 1  # 1: قيمة سيجما الأولى (تمويه خفيف)
blur1 = cv2.GaussianBlur(img, (11, 11), sigma1)  # بنطبق التمويه
# (11, 11): حجم النواة، 1: قيمة سيجما (تمويه خفيف)

sigma2 = 5  # 5: قيمة سيجما التانية (تمويه أقوى)
blur2 = cv2.GaussianBlur(img, (11, 11), sigma2)  # بنطبق التمويه
# (11, 11): حجم النواة، 5: قيمة سيجما (تمويه أقوى)

# حساب الفرق بين الصورتين
diff_img = cv2.subtract(blur1, blur2)  # بنطرح الصورتين علشان نحسب الفرق
# cv2.subtract: بنطرح صورتين من بعض

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
# CIFAR-10: مجموعة بيانات فيها 60,000 صورة ملونة مقسمة لـ 10 فئات

# تقسيم البيانات إلى تدريب واختبار
X_train, X_val, y_train, y_val = train_test_split(X_train[0:5000], y_train[0:5000], test_size=0.1, random_state=42)
# 5000: عدد الصور اللي بنستخدمها، 0.1: نسبة الاختبار (10%)، 42: علشان النتائج تبقى ثابتة

# تسطيح الصور وتطبيعها
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 255: علشان نطبع القيم بين 0 و1
# reshape: بنسطح الصور علشان تبقى متجهات، 255: علشان نطبع القيم
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# إنشاء وتدريب نموذج kNN
knn = KNeighborsClassifier(n_neighbors=5)  # 5: عدد الجيران اللي بنستخدمها في النموذج
# n_neighbors=5: بنستخدم 5 جيران
knn.fit(X_train, y_train)  # بندرب النموذج

# التنبؤ على بيانات الاختبار
y_val_pred = knn.predict(X_val)  # بنتوقع التسميات لبيانات الاختبار
accuracy_test = accuracy_score(y_val, y_val_pred)  # بنحسب الدقة
# accuracy_score: بنحسب دقة النموذج
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
x_train = x_train.astype('float32') / 255.0  # 255: علشان نطبع القيم بين 0 و1
x_test = x_test.astype('float32') / 255.0

# بناء نموذج ConvNet
model = Sequential()  # بنبدأ ننشئ نموذج تسلسلي
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 32: عدد الفلاتر
# Conv2D: طبقة Convolution، 32: عدد الفلاتر، (3, 3): حجم الفلتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64: عدد الفلاتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128: عدد الفلاتر
model.add(MaxPooling2D((2, 2)))  # 2x2: حجم نافذة الـ Max Pooling
model.add(Flatten())  # بنسطح البيانات
model.add(Dense(128, activation='relu'))  # 128: عدد العصبونات في الطبقة الكثيفة
model.add(Dense(10, activation='softmax'))  # 10: عدد العصبونات في الطبقة الأخيرة

# تجميع النموذج
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Adam: مُحسّن Adam، 0.001: معدل التعلم

# تدريب النموذج
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))  # 10: عدد الدورات التدريبية

# تقييم النموذج
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("دقة النموذج على بيانات الاختبار:", test_acc)
