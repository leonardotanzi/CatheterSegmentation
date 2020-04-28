from keras_segmentation.models.unet import resnet50_unet, vgg_unet, mobilenet_unet, unet

# model = mobilenet_unet(n_classes=3, input_height=224, input_width=224)

model = unet(n_classes=3, input_height=416, input_width=608)

model.train(
    train_images="..\\Dataset\\i_30\\Train\\NewTool\\Images",
    train_annotations="..\\Dataset\\i_30\\Train\\NewTool\\Labels",
    checkpoints_path="..\\Checkpoints\\NewTool\\new_unet_tool",
    epochs=5,
    batch_size=4,
    validate=False,
    val_images="..\\Dataset\\i_30\\Validation\\Images",
    val_annotations="..\\Dataset\\i_30\\Validation\\Tool\\Labels",
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