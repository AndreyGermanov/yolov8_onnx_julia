# yolov8_onnx_julia
YOLOv8 inference using Julia

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8) implemented on [Julia](https://julialang.org).

This is a source code for a ["How to create YOLOv8-based object detection web service using Python, Julia, Node.js, JavaScript, Go and Rust"](https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e) tutorial.

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_julia.git`
* Go to the root of cloned repository
* Run the julia REPL: `julia`
* Switch to package installation mode by pressing `]`
* Run `activate .` command
* Run `instantiate` command
* Press `Ctrl+C` to leave the package installation mode
* Exit the REPL: `exit()`

## Run

Execute:

```
julia --project src/object_detector.jl`
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes of all objects detected on it.
