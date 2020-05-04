from keras_segmentation.models.unet import resnet50_unet, vgg_unet, mobilenet_unet, unet

# model = mobilenet_unet(n_classes=3, input_height=224, input_width=224)

model = mobilenet_unet(n_classes=3) # input_height=416, input_width=608)

model.train(
    train_images="..\\..\\UpdatedDataset\\Train\\AllTools\\JPEGImages",
    train_annotations="..\\..\\UpdatedDataset\\Train\\AllTools\\Labels",
    checkpoints_path="..\\..\\Checkpoints\\AllTools\\mobilenet_alltools",
    epochs=5,
    batch_size=4,
    validate=False,
    val_images="..\\..\\UpdatedDataset\\Validation\\AllTools\\JPEGImages",
    val_annotations="..\\..\\UpdatedDataset\\Validation\\AllTools\\Labels",
    steps_per_epoch=32,
    # do_augment=1
)

'''
path = "i_30\\Test (i_15)\\"
for img in tqdm(os.listdir(path)):
    print(img)
    if img.endswith(".png"):
        out = model.predict_segmentation(inp=os.path.join(path, img), out_fname="Output\\{}".format(img))
'''