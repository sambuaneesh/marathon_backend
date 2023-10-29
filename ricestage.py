from roboflow import Roboflow

rf = Roboflow(api_key="kiU6DLRhU8ITt8utfzMi")
project = rf.workspace().project("ricestage")
model = project.version(1).model

# infer on a local image
print(model.predict("100001.jpg").json())

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True).json())

# save an image annotated with your predictions
# model.predict("your_image.jpg").save("prediction.jpg")
