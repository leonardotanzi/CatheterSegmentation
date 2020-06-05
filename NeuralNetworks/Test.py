from keras_segmentation.predict import predict, predict_multiple, model_from_checkpoint_path, evaluate


model = model_from_checkpoint_path("..\\..\\Checkpoints\\AllTools\\mobilenet_alltools")

# BACKGROUND, TOOL, CAT
print(model.evaluate_segmentation(inp_images_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\Original\\JPEGImages\\",
                                  annotations_dir="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\Original\\Labels\\"))

# out = model.predict_segmentation(inp="C:\\Users\\Leonardo\\Desktop\\Catetere\\UpdatedDataset\\Test\\Prova\\JPEGImages\\frame1140.jpg", out_fname="..\\a.png")

'''
predict(checkpoints_path="checkpoints\\vgg_unet_1",
        inp="C:\\Users\\d053175\\Desktop\\Prostate\\Dataset\\Test\\39151.png",
        out_fname="C:\\Users\\d053175\\Desktop\\output.png"
        )


predict_multiple(checkpoints_path="checkpoints\\vgg_unet_1",
		inp_dir="C:\\Users\\d053175\\Desktop\\Prostate\\Test\\",
		out_dir="C:\\Users\\d053175\\Desktop\\outputs\\"
)
'''