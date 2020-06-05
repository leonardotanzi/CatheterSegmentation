from keras_segmentation.models.unet import resnet50_unet, vgg_unet, mobilenet_unet, unet
from keras_segmentation.models.segnet import resnet50_segnet, vgg_segnet, mobilenet_segnet, segnet
from keras_segmentation.models.pspnet import resnet50_pspnet, vgg_pspnet,  pspnet

import os, os.path
from glob import glob

arch = "unet"
model = "mobilenet"

train_images = "..\\..\\UpdatedDataset\\Train\\AllTools\\JPEGImages"
train_annotations = "..\\..\\UpdatedDataset\\Train\\AllTools\\Labels"
val_images = "..\\..\\UpdatedDataset\\Validation\\AllTools\\JPEGImages"
val_annotations = "..\\..\\UpdatedDataset\\Validation\\AllTools\\Labels"
checkpoints_path = "..\\Checkpoints\\{}\\mobilenet".format(arch)

batch_size = 8

n_train = 0
for name in os.listdir(train_images):
    if name.endswith(".jpg"):
        n_train += 1

n_val = 0
for name in os.listdir(val_images):
    if name.endswith(".jpg"):
        n_val += 1

steps_train = n_train // batch_size
steps_val = n_val // batch_size

# {'frequency_weighted_IU': 0.9506827627495642, 'mean_IU': 0.8414459061918275, 'class_wise_IU': array([0.97245269, 0.71134851, 0.84053651])} 5 epoche
# {'frequency_weighted_IU': 0.9400908897672338, 'mean_IU': 0.787267870602049, 'class_wise_IU': array([0.96817914, 0.60285592, 0.79076856])}

# {'frequency_weighted_IU': 0.9586033896038807, 'mean_IU': 0.8684413596796539, 'class_wise_IU': array([0.97701779, 0.73972635, 0.88857994])} 10 epochs
# {'frequency_weighted_IU': 0.9460029601189431, 'mean_IU': 0.8226053892058368, 'class_wise_IU': array([0.96897154, 0.64204238, 0.85680225])}

# {'frequency_weighted_IU': 0.972132136871444, 'mean_IU': 0.9095191231769918, 'class_wise_IU': array([0.98435371, 0.84722829, 0.89697536])} 15 epochs
# {'frequency_weighted_IU': 0.9302741573832257, 'mean_IU': 0.7973039876637206, 'class_wise_IU': array([0.95543   , 0.55825578, 0.87822618])}

# {'frequency_weighted_IU': 0.974522799263118, 'mean_IU': 0.9184853282628952, 'class_wise_IU': array([0.98557866, 0.85710091, 0.91277641])} 20 epochs generalizza
# {'frequency_weighted_IU': 0.897557371103072, 'mean_IU': 0.7474315303247717, 'class_wise_IU': array([0.92634069, 0.43565016, 0.88030374])}

model = mobilenet_unet(n_classes=3) # input_height=416, input_width=608)

model.train(
    train_images=train_images,
    train_annotations=train_annotations,
    checkpoints_path=checkpoints_path,
    epochs=1,
    batch_size=batch_size,
    validate=False,
    val_images=val_images,
    val_annotations=val_annotations,
    steps_per_epoch=steps_train,
    val_steps_per_epoch=steps_val,
    # do_augment=1
)

print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\Original\\JPEGImages\\",
                                  annotations_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\Original\\Labels\\"))

print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\New\\JPEGImages\\",
                                  annotations_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\New\\Labels\\"))

print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\CatCropped\\JPEGImages\\",
                                  annotations_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\CatCropped\\Labels\\"))

print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\PiazzollaCV\\JPEGImages\\",
                                  annotations_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\PiazzollaCV\\Labels\\"))
'''
path = "i_30\\Test (i_15)\\"
for img in tqdm(os.listdir(path)):
    print(img)
    if img.endswith(".png"):
        out = model.predict_segmentation(inp=os.path.join(path, img), out_fname="Output\\{}".format(img))
'''