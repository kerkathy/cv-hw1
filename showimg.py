import cv2

# img_name = "IMG_3147_jpeg_jpg.rf.fc4622004ff72e58b546635774372fe2.jpg"
img_name = "IMG_2489_jpeg_jpg.rf.ffb357957a29cdef43f3fdfb2a13c417.jpg"

img_id = 0
img_path = "./hw1_dataset/train" + "/" + img_name
img = cv2.imread(str(img_path))

# print("value of img[0]: ", img)
print("img_id: ", img_id)
# print("size of img[0]: ", img.size())
# bbox = [
#         [
#                 91,
#                 392,
#                 480,
#                 431
#             ],
#         [
#                 67,
#                 332,
#                 60,
#                 65
#             ],
#             [
#                 615,
#                 117,
#                 140,
#                 70
#             ],



# ]
bbox = [
        [
                0,
                752,
                176,
                243
            ],
            [
                126,
                738,
                431,
                184
            ],
            [
                10,
                845,
                202,
                133
            ],
            [
                490,
                486,
                208,
                123
            ],
            [
                665,
                265,
                102,
                55
            ],
            [
                541,
                933,
                226,
                90
            ]
]


for box in bbox:
        print(box)
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        xmin = x - w/2
        xmax = x + w/2
        ymin = y - h/2
        ymax = y + h/2
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2) 
cv2.imwrite(f"debug-{img_name}.jpg", img)
# DEBUGGNG