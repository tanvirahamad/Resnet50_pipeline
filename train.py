from datetime import datetime
import tensorflow import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from preprocess import data, visulize
from resnet_model import resnt34
EPOCHS = 20
BATCH_SIZE = 24

if __name__ == "__main__":
    res_net_model = resnt34()
    x_train, y_train, x_test, y_test, x_val, y_val = data()

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    checkpoint = ModelCheckpoint(
        "./Model/best_model.h5",
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    res_net_model.fit(x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validaiton_data=(x_val, y_val),
        callbacks=[tensorboard_callback, checkpoint]
    )

    predict_imgs = res_net_model.predict(x_test)

    visulize(10, x_test[:10])
    visulize(10, predict_imgs[:10])