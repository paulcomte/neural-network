# Resources:

- https://keras.io/examples/vision/mnist_convnet/
- https://github.com/tensorflow/tensorflow/issues/33285#issuecomment-2167418423


```mermaid
graph TD
    A["keras.datasets.mnist.load_data() (returns x_train, y_train, x_test, y_test)"] --> B["x_train.astype(float32) / 255, x_test.astype(float32) / 255"]
    B --> C["np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)"]
    C --> D["keras.utils.to_categorical(y_train, num_classes), keras.utils.to_categorical(y_test, num_classes)"]
    D --> E["keras.Sequential([...])"]
    E --> F["layers.Conv2D(32, kernel_size=(3, 3), activation=#quot;relu#quot;)"]
    F --> G["layers.MaxPooling2D(pool_size=(2, 2))"]
    G --> H["layers.Conv2D(64, kernel_size=(3, 3), activation=#quot;relu#quot;)"]
    H --> I["layers.MaxPooling2D(pool_size=(2, 2))"]
    I --> J["layers.Flatten()"]
    J --> K["layers.Dropout(0.5)"]
    K --> L["layers.Dense(num_classes, activation=#quot;softmax#quot;)"]
    L --> M["model.compile(loss=#quot;categorical_crossentropy#quot;, optimizer=#quot;adam#quot;, metrics=[#quot;accuracy#quot;])"]
    M --> N["model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)"]
    N --> O["model.evaluate(x_test, y_test, verbose=0)"]
    O --> P[Display Test Accuracy]

    subgraph Model_Building [Build Model Function]
        direction TB
        E --> F
        F --> G
        G --> H
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
    end

    subgraph Train_Model [Train Model Function]
        direction TB
        N --> O
    end

    subgraph Evaluate_Model [Evaluate Model Function]
        direction TB
        O --> P
    end
  ```
