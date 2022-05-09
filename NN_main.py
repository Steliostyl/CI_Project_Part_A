import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import gradient_descent_v2
from keras.regularizers import L2
from data_parsing import parse_data, parse_label
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Used for the graphs. Takes as input the name of the metric and 
# outputs graphs with the progression of said metric over EPOCHS.
def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)

H = 0.05            # Learning rate.
NW_IN = 8520        # Input of the network. Equal to the ammount of available words
NW_OUT = 20         # Output of the network (labels). Equal to the ammount of categories
EPOCHS = 50         # Epochs in training
BATCH_SIZE = 12     # Batch size

# First, we parse our input data and labels into an np.array
# using the functions parse_data and parse_label

train_data = parse_data("train-data.dat")
train_label = parse_label("train-label.dat")

test_data = parse_data("test-data.dat")
test_label = parse_label("test-label.dat")

# Callback function. Used for early stopping in training. 
# This is different for every test after inspecting the plotted graphs.
callback_function = keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy", patience=3, min_delta=0.0005 , mode="max"
)

model_results = []
max_accuracy = 0

for i, (train_index, test_index) in enumerate(kfold.split(train_data)):    
    # Create model
    model = Sequential()
    model.add(Dense((NW_IN + NW_OUT), activation="relu"))                                     # Hidden Layer 1
    model.add(Dense((NW_IN + NW_OUT)/2, activation="relu", kernel_regularizer=L2(l2=0.1)))  # Hidden Layer 2
    model.add(Dense(20 , activation="sigmoid"))                                             # Output Layer
    # Compile the model with desired parameters (loss function, optimizer, metrics to keep track of etc.)
    model.compile(
        loss="binary_crossentropy",
        optimizer=gradient_descent_v2.SGD(learning_rate=H, momentum=0.6),
        metrics=["MSE","accuracy", "binary_accuracy"]
    )

    # Train model using KFOLD
    history = model.fit(
        train_data[train_index],    # Input for training
        train_label[train_index],   # Labels for training
        validation_data=(train_data[test_index], train_label[test_index]), # Validation data for KFOLD
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[callback_function]
    )

    # Evaluate trained model and save the desired metrics (defined in model.compile)
    loss, mse, acc, binary_acc = model.evaluate(train_data[test_index], train_label[test_index])
    model_results.append((loss, mse, acc, binary_acc))

    # If current model has the best accuracy, save it as the best model
    if binary_acc > max_accuracy:
        best_model = model
        best_history = history

# Print results for each test (fold)
for i, result in enumerate(model_results):
    print(result[3])
    print(f"Fold number {i+1}. Test accuracy: {round(result[3] * 100, 2)}%.")

# Evaluate the best model derived from previous KFOLD training on our real test data
train_loss, train_mse, train_acc, train_bin_acc = best_model.evaluate(train_data, train_label, verbose=0)
test_loss, test_mse, test_acc, test_bin_acc = best_model.evaluate(test_data, test_label, verbose=0)

# Print the validation results of the best model
print(f"Train accuracy: {round(train_acc * 100, 2)}%, Test accuracy: {round(test_acc * 100, 2)}%")
print(f"Train binary accuracy: {round(train_bin_acc * 100, 2)}%, Test binary accuracy: {round(test_bin_acc * 100, 2)}%")
print("Train MSE:",train_mse,"Test MSE:", test_mse)
print("Train loss:",train_loss,"Test loss:", test_loss)

# Plot the results
plot_result("loss")   
plot_result("accuracy")
plot_result("binary_accuracy")