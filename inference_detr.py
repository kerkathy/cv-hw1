# from transformers import DetrFeatureExtractor
from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import torch # put this line after `... import DetrFeatureExtractor` or else some stdc++ error occurs

model_path = "detr-object-detection-finetuned/"
image = Image.open('hw1_dataset/images/test/IMG_2574_jpeg_jpg.rf.ca0c3ad32384309a61e92d9a8bef87b9.jpg')

model = DetrForObjectDetection.from_pretrained(model_path)
processor = DetrImageProcessor.from_pretrained(model_path)
encoding = processor(image, return_tensors="pt")

# Visualize the result
with torch.no_grad():
    outputs = model(**encoding)

width, height = image.size

postprocessed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.5)

results = postprocessed_outputs[0]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, scores, labels, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
            text = f'{model.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig('test_img.png')
        print("Save output img to test_img.png")

plot_results(image, results['scores'], results['labels'], results['boxes'])



